"""
The routing orchestrator: manages state transitions between the three
specialized agents and owns the Model Context Protocol session used by
the Fraud Detection stage.

This is the file that answers "how does the multi-agent workflow actually
run end to end" — see the companion explanation document for the full
walkthrough of why it's structured as explicit Python control flow rather
than an ADK `SequentialAgent` graph (short version: our Compliance stage
is a HYBRID of a deterministic rule engine and an LLM narrative step,
which doesn't fit a pure LLM-agent pipeline, and explicit control flow
makes it trivial to log a clean, ordered "state transition" event before
and after every stage for the UI's live terminal panel).
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools.mcp_tool import McpToolset, StdioConnectionParams
from google.genai import types
from mcp import StdioServerParameters

from src.agents import (
    ComplianceExplanation,
    ComplianceRuleEngine,
    DocumentAnalysisResult,
    FraudDetectionResult,
    RuleEngineVerdict,
    build_compliance_explanation_agent,
    build_compliance_explanation_prompt,
    build_document_analysis_agent,
    build_document_analysis_prompt,
    build_fraud_detection_agent,
    build_fraud_detection_prompt,
    compute_document_quality_passed,
)
from src.guardrails_and_logging import (
    clear_run,
    get_logger,
    get_run_logs,
    start_run,
    validate_json_output,
)
from src.mock_data_generator import ApplicantCase

log = get_logger("orchestrator")

MCP_SERVER_SCRIPT = Path(__file__).resolve().parent / "mcp_tools_server.py"


@dataclass
class TriageResult:
    case_id: str
    run_id: str
    document_result: DocumentAnalysisResult
    identity_comparison: dict
    watchlist_check: dict
    fraud_result: FraudDetectionResult
    document_quality_passed: bool
    verdict: RuleEngineVerdict
    explanation: str
    log_events: list[dict] = field(default_factory=list)

    def to_case_file_dict(self) -> dict:
        """Everything needed for the downloadable audit/case-file JSON that
        the Streamlit UI offers for download — this is what makes agent
        decisions "traceable" rather than just a black-box verdict.
        """
        return {
            "case_id": self.case_id,
            "run_id": self.run_id,
            "decision": self.verdict.decision,
            "decision_reason": self.verdict.reason,
            "matched_rule": self.verdict.matched_rule,
            "explanation": self.explanation,
            "document_analysis": self.document_result.model_dump(),
            "identity_comparison": self.identity_comparison,
            "watchlist_check": self.watchlist_check,
            "fraud_detection": self.fraud_result.model_dump(),
            "document_quality_passed": self.document_quality_passed,
            "audit_log": self.log_events,
        }


async def _run_llm_agent(agent: LlmAgent, prompt: str, *, app_name: str) -> str:
    """Run one LlmAgent for a single turn and return its final text reply.

    Each pipeline stage gets its own fresh InMemoryRunner + session rather
    than sharing one long conversation. That costs a little redundant
    session-setup overhead, but keeps every stage isolated, independently
    testable, and easy to log — worth it at this app's scale (a handful of
    LLM calls per triage run, not a high-throughput service).
    """
    runner = InMemoryRunner(agent=agent, app_name=app_name)
    try:
        session = await runner.session_service.create_session(app_name=app_name, user_id="triage")
        final_text = ""
        async for event in runner.run_async(
            user_id="triage",
            session_id=session.id,
            new_message=types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        final_text = part.text
        return final_text
    finally:
        await runner.close()


def _unwrap_mcp_tool_result(raw: dict) -> dict:
    """ADK's MCP tool wrapper returns the raw MCP content envelope, e.g.
    `{"content": [{"type": "text", "text": "<json string>"}], "isError": false}`.
    Unwrap it to the plain dict our tool functions actually returned.
    """
    if raw.get("isError"):
        raise RuntimeError(f"MCP tool call failed: {raw}")
    text = raw["content"][0]["text"]
    return json.loads(text)


async def _call_mcp_tools(document: dict, history: dict | None, run_id: str) -> tuple[dict, dict]:
    """Spawn the MCP tool server as a child process, call both
    cross-reference tools over the real MCP protocol, then shut it down.
    This is the only place in the codebase that speaks MCP directly.
    """
    toolset = McpToolset(
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(command=sys.executable, args=[str(MCP_SERVER_SCRIPT)]),
            timeout=15.0,
        )
    )
    try:
        tools = {tool.name: tool for tool in await toolset.get_tools()}

        log.info("mcp_tool_invocation_started", run_id=run_id, tool="compare_identity_fields")
        identity_comparison = _unwrap_mcp_tool_result(
            await tools["compare_identity_fields"].run_async(
                args={"document": document, "history": history}, tool_context=None
            )
        )
        log.info(
            "mcp_tool_invocation_completed",
            run_id=run_id,
            tool="compare_identity_fields",
            result=identity_comparison,
        )

        log.info("mcp_tool_invocation_started", run_id=run_id, tool="check_watchlist_and_flags")
        watchlist_check = _unwrap_mcp_tool_result(
            await tools["check_watchlist_and_flags"].run_async(
                args={"history": history}, tool_context=None
            )
        )
        log.info(
            "mcp_tool_invocation_completed",
            run_id=run_id,
            tool="check_watchlist_and_flags",
            result=watchlist_check,
        )

        return identity_comparison, watchlist_check
    finally:
        await toolset.close()


_CONTRADICTION_WORDS: dict[str, list[str]] = {
    "approve": ["reject", "denied", "decline"],
    "reject": ["approve", "approved"],
    "manual_review": [],
}


def _guard_explanation_matches_decision(explanation: str, verdict: RuleEngineVerdict, run_id: str) -> str:
    """Guardrail: the narrative LLM was explicitly told not to contradict
    the rule engine's decision. This is the safety net in case it does
    anyway — falls back to the policy engine's own (trusted) reason text.
    """
    lowered = explanation.lower()
    if any(word in lowered for word in _CONTRADICTION_WORDS.get(verdict.decision, [])):
        log.warning(
            "guardrail_explanation_contradicts_decision",
            run_id=run_id,
            decision=verdict.decision,
            explanation_preview=explanation[:200],
        )
        return verdict.reason
    return explanation


async def run_triage_async(case: ApplicantCase, run_id: str | None = None) -> TriageResult:
    run_id = run_id or str(uuid.uuid4())
    start_run(run_id)
    document = case.document.to_dict()
    history = case.history.to_dict() if case.history else None

    log.info("triage_run_started", run_id=run_id, case_id=case.case_id, scenario=case.scenario)

    # --- Stage 1: Document Analysis ---
    log.info("agent_invocation_started", run_id=run_id, agent="document_analysis_agent")
    doc_prompt = build_document_analysis_prompt(json.dumps(document, indent=2))
    doc_raw_text = await _run_llm_agent(
        build_document_analysis_agent(), doc_prompt, app_name="kyc_document_analysis"
    )
    document_result = validate_json_output(
        doc_raw_text, DocumentAnalysisResult, agent_name="document_analysis_agent"
    ) or DocumentAnalysisResult(
        data_quality_ok=False,
        quality_issues=["Agent output failed guardrail validation; treated as low quality."],
        normalized_full_name=document["full_name"],
        summary="Document analysis output could not be validated; flagged for manual review.",
    )
    document_quality_passed = compute_document_quality_passed(document)
    log.info(
        "agent_invocation_completed",
        run_id=run_id,
        agent="document_analysis_agent",
        result=document_result.model_dump(),
        deterministic_quality_gate_passed=document_quality_passed,
    )

    # --- Stage 2: MCP tool calls (deterministic cross-referencing) ---
    identity_comparison, watchlist_check = await _call_mcp_tools(document, history, run_id)

    # --- Stage 3: Fraud & Anomaly Detection ---
    log.info("agent_invocation_started", run_id=run_id, agent="fraud_detection_agent")
    fraud_prompt = build_fraud_detection_prompt(
        json.dumps(document, indent=2),
        json.dumps(identity_comparison, indent=2),
        json.dumps(watchlist_check, indent=2),
    )
    fraud_raw_text = await _run_llm_agent(
        build_fraud_detection_agent(), fraud_prompt, app_name="kyc_fraud_detection"
    )
    fraud_result = validate_json_output(
        fraud_raw_text, FraudDetectionResult, agent_name="fraud_detection_agent"
    ) or FraudDetectionResult(
        risk_score=100,
        risk_flags=["llm_output_invalid_escalate"],
        reasoning="Fraud agent output failed guardrail validation; escalating to maximum risk out of caution.",
    )
    log.info(
        "agent_invocation_completed",
        run_id=run_id,
        agent="fraud_detection_agent",
        result=fraud_result.model_dump(),
    )

    # --- Stage 4: Deterministic compliance rule engine ---
    verdict = ComplianceRuleEngine().evaluate(
        risk_score=fraud_result.risk_score,
        is_on_watchlist=watchlist_check["is_on_watchlist"],
        document_quality_passed=document_quality_passed,
    )
    log.info("rule_engine_decision", run_id=run_id, verdict=verdict.model_dump())

    # --- Stage 5: Compliance explanation (LLM narrative only, decision already final) ---
    log.info("agent_invocation_started", run_id=run_id, agent="compliance_explanation_agent")
    explanation_prompt = build_compliance_explanation_prompt(verdict, fraud_result, document_result)
    explanation_raw_text = await _run_llm_agent(
        build_compliance_explanation_agent(), explanation_prompt, app_name="kyc_compliance_explanation"
    )
    explanation_result = validate_json_output(
        explanation_raw_text, ComplianceExplanation, agent_name="compliance_explanation_agent"
    )
    explanation = _guard_explanation_matches_decision(
        explanation_result.explanation if explanation_result else verdict.reason, verdict, run_id
    )
    log.info(
        "agent_invocation_completed",
        run_id=run_id,
        agent="compliance_explanation_agent",
        explanation=explanation,
    )

    log.info("triage_run_completed", run_id=run_id, case_id=case.case_id, decision=verdict.decision)

    result = TriageResult(
        case_id=case.case_id,
        run_id=run_id,
        document_result=document_result,
        identity_comparison=identity_comparison,
        watchlist_check=watchlist_check,
        fraud_result=fraud_result,
        document_quality_passed=document_quality_passed,
        verdict=verdict,
        explanation=explanation,
        log_events=get_run_logs(run_id),
    )
    clear_run(run_id)
    return result


def run_triage(case: ApplicantCase, run_id: str | None = None) -> TriageResult:
    """Synchronous entry point for Streamlit, which is not async-native.

    Every function above this one is `async` because ADK's Runner and the
    MCP client are both built on asyncio. `asyncio.run()` here is the one
    bridge point between Streamlit's sync world and ADK/MCP's async world.

    Accepts an optional pre-generated `run_id` so a caller (app.py) can
    call `guardrails_and_logging.subscribe_to_run(run_id, ...)` BEFORE
    starting the run, in a background thread, to get genuinely live log
    events instead of only seeing them after the whole run completes.
    """
    return asyncio.run(run_triage_async(case, run_id=run_id))
