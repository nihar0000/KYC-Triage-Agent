"""
Tests for src/mock_data_generator.py.

These are the easiest tests in the project to understand if you're new to
pytest: each `test_*` function is a self-contained check. Run them all
with `pytest` from the project root, or just this file with
`pytest tests/test_mock_data_generator.py -v`.
"""

from src.mock_data_generator import (
    generate_applicant_case,
    generate_customer_history,
    generate_extracted_document,
)


def test_same_seed_produces_same_document():
    """Determinism check: this is the property that makes 'upload the same
    file twice, get the same simulated identity' possible in the UI.
    """
    doc_a = generate_extracted_document(seed="abc123")
    doc_b = generate_extracted_document(seed="abc123")
    assert doc_a.to_dict() == doc_b.to_dict()


def test_different_seeds_produce_different_documents():
    doc_a = generate_extracted_document(seed="seed-one")
    doc_b = generate_extracted_document(seed="seed-two")
    assert doc_a.full_name != doc_b.full_name


def test_clean_scenario_all_identity_fields_match():
    """Regression test: an earlier version only aligned the name, leaving
    DOB/address to mismatch by chance — which would make a real LLM flag
    a supposedly-clean case as risky, since the Fraud Agent is explicitly
    instructed to treat those mismatches as risk signals.
    """
    case = generate_applicant_case(seed=1, scenario="clean")
    assert case.history is not None
    assert case.document.full_name == case.history.full_name_on_file
    assert case.document.date_of_birth == case.history.date_of_birth_on_file
    assert case.document.address == case.history.address_on_file


def test_watchlist_hit_scenario_identity_fields_still_match():
    """The watchlist scenario should isolate the watchlist flag as the
    only anomaly — identity fields should line up just like "clean".
    """
    case = generate_applicant_case(seed=1, scenario="watchlist_hit")
    assert case.history is not None
    assert case.document.full_name == case.history.full_name_on_file
    assert case.document.date_of_birth == case.history.date_of_birth_on_file


def test_fraud_mismatch_scenario_names_differ():
    case = generate_applicant_case(seed=1, scenario="fraud_mismatch")
    assert case.history is not None
    assert case.document.full_name != case.history.full_name_on_file


def test_watchlist_scenario_forces_watchlist_flag():
    case = generate_applicant_case(seed=1, scenario="watchlist_hit")
    assert case.history is not None
    assert case.history.is_on_watchlist is True


def test_incomplete_scenario_has_no_history_and_low_confidence():
    case = generate_applicant_case(seed=1, scenario="incomplete")
    assert case.history is None
    assert case.document.extraction_confidence < 0.6


def test_customer_history_watchlist_flag_matches_prior_flags():
    history = generate_customer_history(seed=99, force_watchlist=True)
    assert history.is_on_watchlist is True
