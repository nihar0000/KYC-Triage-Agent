"""
Picks which LLM actually powers agent reasoning, and provides a fully
offline fallback so the public demo never breaks for a visitor who
hasn't set up an API key.

Why this file exists as its own module: agents.py and orchestrator.py
should not need to know or care whether we're calling Gemini, Groq, or
running in offline mock mode. They just ask `get_model()` for "a model"
and pass it straight to `google.adk.agents.LlmAgent(model=...)`, which
accepts either a plain model-name string OR a `BaseLlm` object.

Provider priority (checked in this order):
  1. GEMINI_API_KEY / GOOGLE_API_KEY set -> use Gemini directly.
  2. GROQ_API_KEY set -> use Groq, via ADK's LiteLLM integration.
  3. Neither set -> MockLlm (fully offline, deterministic-ish, canned
     JSON responses shaped to look like a real agent's output).

Why Gemini uses a plain string but Groq needs an extra wrapper:
`google-adk` has NATIVE, first-class support for Gemini models (it calls
the `google-genai` SDK internally when you pass a string like
"gemini-2.0-flash"). Every other provider — Groq, OpenAI, Anthropic,
100+ others — goes through LiteLLM, a separate library ADK optionally
depends on (installed via the `google-adk[extensions]` extra). Passing a
string like "groq/llama-3.3-70b-versatile" makes ADK auto-resolve it to
`LiteLlm(model="groq/llama-3.3-70b-versatile")` behind the scenes, which
then reads GROQ_API_KEY from the environment itself (LiteLLM's own
convention, not something we handle manually).

Alternative approach considered and rejected: writing a hand-rolled
`GroqLlm(BaseLlm)` subclass that calls the `groq` Python SDK directly
(no LiteLLM). This would be a lighter dependency (no litellm/openai/
tiktoken pulled in), but it would also mean re-implementing ADK's
request/response translation AND its function-calling (tool call)
translation ourselves — and getting that translation subtly wrong would
silently break the Fraud Detection Agent's MCP tool calls under Groq
while looking fine under Gemini. Given this is meant to be
production-ready, we pay the heavier install for the officially
supported, tested code path instead.
"""

from __future__ import annotations

import json
import os
import re
from typing import AsyncGenerator

from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types

from src.guardrails_and_logging import get_logger

log = get_logger("llm_factory")

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def active_provider() -> str:
    """Report which provider is active, for display in the UI's sidebar."""
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return f"gemini ({DEFAULT_GEMINI_MODEL})"
    if os.getenv("GROQ_API_KEY"):
        return f"groq ({DEFAULT_GROQ_MODEL})"
    return "offline mock (no API key set)"


def get_model(role: str) -> "str | MockLlm":
    """Return whatever should be passed as `LlmAgent(model=...)`.

    `role` is one of "document_analysis", "fraud_detection", or
    "compliance_explanation" — it only matters for MockLlm, so its canned
    answers are shaped correctly for whichever agent is asking.
    """
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        # google-genai (used internally by ADK's native Gemini support)
        # specifically reads GOOGLE_API_KEY, so normalize GEMINI_API_KEY
        # into it if that's the one the user/HF Space secret set.
        os.environ.setdefault("GOOGLE_API_KEY", gemini_key)
        return DEFAULT_GEMINI_MODEL

    if os.getenv("GROQ_API_KEY"):
        return f"groq/{DEFAULT_GROQ_MODEL}"

    log.warning("llm_offline_mock_mode_active", role=role)
    return MockLlm(model=f"mock-llm-{role}", role=role)


class MockLlm(BaseLlm):
    """A zero-dependency, zero-API-key stand-in for a real LLM.

    This is NOT trying to simulate real language understanding — it looks
    for a few structured signals already present in the prompt (which our
    agents always embed as compact JSON, see agents.py) and returns a
    plausible, schema-valid JSON response using simple `if` logic. This
    keeps the public demo fully functional (every screen renders, every
    decision path is reachable) for a visitor who hasn't configured any
    API key, while being 100% honest in the UI that it's not a real model
    (see app.py's "offline demo mode" banner).

    Note: MockLlm intentionally does NOT support tool/function calling.
    The Fraud Detection Agent's MCP tool calls require a real LLM that
    can decide to call a tool — see orchestrator.py for how we handle
    that gap (we call the MCP tools directly, in Python, before invoking
    MockLlm, and hand it the results as plain text instead).
    """

    role: str = "generic"

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        prompt_text = _flatten_request_text(llm_request)
        reply_json = _mock_reply_for(self.role, prompt_text)
        yield LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=json.dumps(reply_json, indent=2))],
            ),
            turn_complete=True,
        )


def _flatten_request_text(llm_request: LlmRequest) -> str:
    chunks: list[str] = []
    for content in llm_request.contents:
        for part in content.parts or []:
            if part.text:
                chunks.append(part.text)
    return "\n".join(chunks)


def _mock_reply_for(role: str, prompt_text: str) -> dict:
    """Cheap heuristic 'reasoning' so the offline demo still branches
    sensibly instead of always returning the same canned answer.
    """
    text_lower = prompt_text.lower()

    if role == "document_analysis":
        low_confidence = '"extraction_confidence": 0.3' in prompt_text or bool(
            re.search(r'"extraction_confidence":\s*0\.[0-5]', prompt_text)
        )
        missing_expiry = '"expiration_date": null' in prompt_text
        return {
            "data_quality_ok": not (low_confidence or missing_expiry),
            "quality_issues": (
                ["low OCR extraction confidence"] if low_confidence else []
            )
            + (["missing/expired document expiration"] if missing_expiry else []),
            "normalized_full_name": _extract_json_field(prompt_text, "full_name") or "Unknown",
            "summary": "[offline mock] Document parsed using heuristic rules, not a real LLM.",
        }

    if role == "fraud_detection":
        watchlist_hit = '"is_on_watchlist": true' in text_lower
        name_mismatch = '"names_match": false' in text_lower
        return {
            "risk_score": 90 if watchlist_hit else (65 if name_mismatch else 12),
            "risk_flags": (["watchlist_hit"] if watchlist_hit else [])
            + (["name_mismatch"] if name_mismatch else []),
            "reasoning": "[offline mock] Risk score derived from heuristic rules, not a real LLM.",
        }

    if role == "compliance_explanation":
        decision_match = re.search(r"already made by the policy engine:\s*(\w+)", prompt_text)
        reason_match = re.search(r"Policy reason:\s*(.+)", prompt_text)
        decision = decision_match.group(1) if decision_match else "manual_review"
        reason = reason_match.group(1).strip() if reason_match else "policy thresholds were applied"
        return {
            "explanation": (
                f"[offline mock] Based on the policy engine's {decision} decision "
                f"({reason}), this case file has been recorded accordingly. "
                "(This narrative was generated by heuristic rules, not a real LLM.)"
            )
        }

    return {"summary": "[offline mock] No API key configured for this demo session."}


def _extract_json_field(text: str, field: str) -> str | None:
    match = re.search(rf'"{field}":\s*"([^"]*)"', text)
    return match.group(1) if match else None
