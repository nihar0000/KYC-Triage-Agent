"""
End-to-end integration tests for src/orchestrator.py.

These run the FULL pipeline — Document Analysis -> real MCP tool calls
(a real child process, talking real MCP protocol) -> Fraud Detection ->
the deterministic rule engine -> Compliance Explanation — with no network
access required, by forcing offline mock-LLM mode via monkeypatch. This
is what proves the whole system holds together end to end, not just its
individual pieces.

These are slower than the unit tests (spawning a subprocess per test
takes a moment) — that's expected and fine for an integration test.
"""

from src.mock_data_generator import generate_applicant_case
from src.orchestrator import run_triage


def _force_offline_mode(monkeypatch):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_clean_scenario_is_approved(monkeypatch):
    _force_offline_mode(monkeypatch)
    case = generate_applicant_case(seed=1, scenario="clean")
    result = run_triage(case)
    assert result.verdict.decision == "approve"
    assert result.document_quality_passed is True
    assert result.watchlist_check["is_on_watchlist"] is False


def test_watchlist_hit_scenario_is_rejected(monkeypatch):
    _force_offline_mode(monkeypatch)
    case = generate_applicant_case(seed=2, scenario="watchlist_hit")
    result = run_triage(case)
    assert result.verdict.decision == "reject"
    assert result.watchlist_check["is_on_watchlist"] is True


def test_incomplete_scenario_goes_to_manual_review(monkeypatch):
    _force_offline_mode(monkeypatch)
    case = generate_applicant_case(seed=3, scenario="incomplete")
    result = run_triage(case)
    assert result.verdict.decision == "manual_review"
    assert result.document_quality_passed is False


def test_fraud_mismatch_scenario_flags_name_mismatch(monkeypatch):
    _force_offline_mode(monkeypatch)
    case = generate_applicant_case(seed=4, scenario="fraud_mismatch")
    result = run_triage(case)
    assert result.identity_comparison["names_match"] is False
    # Decision itself depends on the heuristic mock's risk score, but the
    # mismatch must always be visible in the audit trail regardless.
    assert "name_mismatch" in result.fraud_result.risk_flags


def test_case_file_export_contains_full_audit_trail(monkeypatch):
    _force_offline_mode(monkeypatch)
    case = generate_applicant_case(seed=5, scenario="clean")
    result = run_triage(case)
    case_file = result.to_case_file_dict()
    assert case_file["decision"] == result.verdict.decision
    assert len(case_file["audit_log"]) > 0
    assert any(event["event"] == "triage_run_completed" for event in case_file["audit_log"])
