"""
Tests for the MCP tool functions in src/mcp_tools_server.py.

Note what we're NOT doing here: spinning up the MCP server as a
subprocess and talking to it over the real protocol. That's exercised
manually (and in orchestrator.py at runtime) but would make these tests
slow and flaky in CI. Because every `@mcp.tool()`-decorated function is
still a perfectly ordinary Python function underneath, we can call it
directly and test its actual logic in isolation — the MCP decorator only
adds protocol plumbing on top, it doesn't change what the function does.
"""

from src.mcp_tools_server import check_watchlist_and_flags, compare_identity_fields


def test_compare_identity_fields_new_customer_has_no_history():
    result = compare_identity_fields(document={"full_name": "Jane Doe"}, history=None)
    assert result["has_history_on_file"] is False
    assert result["is_new_customer"] is True


def test_compare_identity_fields_matching_names():
    document = {"full_name": "Jane Doe", "date_of_birth": "1990-01-01", "address": "1 Main St, Toronto"}
    history = {
        "full_name_on_file": "Jane Doe",
        "date_of_birth_on_file": "1990-01-01",
        "address_on_file": "1 Main St, Toronto",
    }
    result = compare_identity_fields(document=document, history=history)
    assert result["names_match"] is True
    assert result["dob_match"] is True
    assert result["address_match"] is True


def test_compare_identity_fields_mismatched_names():
    document = {"full_name": "Jane Doe", "date_of_birth": "1990-01-01", "address": "1 Main St"}
    history = {
        "full_name_on_file": "Robert Smith",
        "date_of_birth_on_file": "1985-06-15",
        "address_on_file": "99 Other Ave",
    }
    result = compare_identity_fields(document=document, history=history)
    assert result["names_match"] is False
    assert result["dob_match"] is False


def test_compare_identity_fields_tolerates_minor_name_differences():
    document = {"full_name": "Jon  Smith", "date_of_birth": "1990-01-01", "address": "1 Main St"}
    history = {
        "full_name_on_file": "Jon Smith",
        "date_of_birth_on_file": "1990-01-01",
        "address_on_file": "1 Main St",
    }
    result = compare_identity_fields(document=document, history=history)
    assert result["names_match"] is True  # extra whitespace shouldn't break the match


def test_check_watchlist_and_flags_new_customer():
    result = check_watchlist_and_flags(history=None)
    assert result == {"is_on_watchlist": False, "prior_flags": [], "flag_severity": 0}


def test_check_watchlist_and_flags_watchlist_hit_dominates_severity():
    history = {"is_on_watchlist": True, "prior_flags": ["sanctions_screening_hit"]}
    result = check_watchlist_and_flags(history=history)
    assert result["is_on_watchlist"] is True
    assert result["flag_severity"] >= 50


def test_check_watchlist_and_flags_severity_caps_at_100():
    history = {"is_on_watchlist": True, "prior_flags": ["a", "b", "c", "d", "e", "f", "g", "h"]}
    result = check_watchlist_and_flags(history=history)
    assert result["flag_severity"] == 100
