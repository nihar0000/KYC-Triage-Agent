"""
Tests for src/agents.py — specifically the deterministic parts: the
Compliance Rule Engine and the document-quality gate. These are the two
functions in the whole codebase where "correct" has a precise, testable
meaning (unlike LLM output, which is inherently fuzzy).
"""

import pytest

from src.agents import ComplianceRuleEngine, compute_document_quality_passed


@pytest.fixture()
def rule_engine() -> ComplianceRuleEngine:
    return ComplianceRuleEngine()


def test_watchlist_hit_always_rejects_regardless_of_risk_score(rule_engine):
    verdict = rule_engine.evaluate(risk_score=0, is_on_watchlist=True, document_quality_passed=True)
    assert verdict.decision == "reject"
    assert verdict.matched_rule == "hard_stop:watchlist_hit"


def test_failed_document_quality_forces_manual_review(rule_engine):
    verdict = rule_engine.evaluate(risk_score=0, is_on_watchlist=False, document_quality_passed=False)
    assert verdict.decision == "manual_review"
    assert verdict.matched_rule == "hard_stop:document_quality_failed"


def test_low_risk_score_approves(rule_engine):
    verdict = rule_engine.evaluate(risk_score=10, is_on_watchlist=False, document_quality_passed=True)
    assert verdict.decision == "approve"


def test_mid_risk_score_goes_to_manual_review(rule_engine):
    verdict = rule_engine.evaluate(risk_score=50, is_on_watchlist=False, document_quality_passed=True)
    assert verdict.decision == "manual_review"


def test_high_risk_score_rejects(rule_engine):
    verdict = rule_engine.evaluate(risk_score=90, is_on_watchlist=False, document_quality_passed=True)
    assert verdict.decision == "reject"


def test_watchlist_hard_stop_takes_priority_over_low_risk_score(rule_engine):
    """The whole point of a hard stop: a low fraud score must not override it."""
    verdict = rule_engine.evaluate(risk_score=5, is_on_watchlist=True, document_quality_passed=True)
    assert verdict.decision == "reject"


def test_document_quality_fails_on_low_confidence():
    document = {"extraction_confidence": 0.4, "document_type": "utility_bill", "expiration_date": None}
    assert compute_document_quality_passed(document) is False


def test_document_quality_fails_on_missing_expiry_for_id_documents():
    document = {"extraction_confidence": 0.9, "document_type": "passport", "expiration_date": None}
    assert compute_document_quality_passed(document) is False


def test_document_quality_passes_for_utility_bill_without_expiry():
    document = {"extraction_confidence": 0.9, "document_type": "utility_bill", "expiration_date": None}
    assert compute_document_quality_passed(document) is True


def test_document_quality_passes_for_good_id_document():
    document = {"extraction_confidence": 0.9, "document_type": "passport", "expiration_date": "2030-01-01"}
    assert compute_document_quality_passed(document) is True
