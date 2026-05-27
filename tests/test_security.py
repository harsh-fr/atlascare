"""
tests/test_security.py
=======================
Security and ownership enforcement tests.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from tests.conftest import make_tool_mock, make_done_mock, make_text_mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(client, message, session_id="sess-cust001"):
    resp = client.post("/query", json={"message": message, "session_id": session_id})
    try:
        body = resp.json()
    except Exception:
        body = {}
    return {"status": resp.status_code, "body": body}


def _run(client, message, groq_responses, session_id="sess-cust001"):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=groq_responses)
    with patch("agent.graph._groq_client", mock_client):
        return _post(client, message, session_id)


def _statuses(result): return [tc["status"] for tc in result["body"].get("trace", {}).get("tool_calls", [])]
def _actions(result):  return [tc["action"] for tc in result["body"].get("trace", {}).get("tool_calls", [])]


# ===========================================================================
# Cross-customer access denial
# ===========================================================================

class TestCrossCustomerAccess:

    def test_cust001_cannot_access_cust002_order(self, client):
        result = _run(client, "Where is order ORD-99001?", [
            make_tool_mock("get_order", {"order_id": "ORD-99001"}),
            make_text_mock("I couldn't find that order."),
        ])
        assert result["status"] == 200
        assert "ownership_denied" in _statuses(result)

    def test_cust001_cannot_cancel_cust002_item(self, client):
        result = _run(client, "Cancel item 1 from ORD-99001.", [
            make_tool_mock("cancel_item", {"order_id": "ORD-99001", "line_id": 1}),
            make_text_mock("I couldn't find that order."),
        ])
        assert result["status"] == 200
        assert "ownership_denied" in _statuses(result)

    def test_cust001_cannot_refund_cust002_order(self, client):
        result = _run(client, "Refund ORD-99001.", [
            make_tool_mock("process_refund", {"order_id": "ORD-99001", "amount_inr": 1000.0, "method": "HDFC_CREDIT"}),
            make_text_mock("I couldn't find that order."),
        ])
        assert result["status"] == 200
        assert "ownership_denied" in _statuses(result)

    def test_cust001_cannot_update_address_of_cust002_order(self, client):
        result = _run(client, "Update address for ORD-99001.", [
            make_tool_mock("update_address", {"order_id": "ORD-99001", "address_label": "home"}),
            make_text_mock("I couldn't find that order."),
        ])
        assert result["status"] == 200
        assert "ownership_denied" in _statuses(result)

    def test_ownership_error_message_is_vague(self, client):
        result = _run(client, "Track ORD-99001.", [
            make_tool_mock("get_order", {"order_id": "ORD-99001"}),
            make_text_mock("I couldn't find that order in your account."),
        ])
        resp = result["body"].get("response", "").lower()
        assert "cust-002"         not in resp
        assert "another customer" not in resp
        assert "belongs to"       not in resp

    def test_nonexistent_order_same_response_as_other_customer_order(self, client):
        r1 = _run(client, "Track ORD-00000.", [
            make_tool_mock("get_order", {"order_id": "ORD-00000"}),
            make_text_mock("Order not found."),
        ])
        r2 = _run(client, "Track ORD-99001.", [
            make_tool_mock("get_order", {"order_id": "ORD-99001"}),
            make_text_mock("Order not found."),
        ])
        s1 = [tc["status"] for tc in r1["body"].get("trace", {}).get("tool_calls", [])
              if tc["action"] == "get_order"]
        s2 = [tc["status"] for tc in r2["body"].get("trace", {}).get("tool_calls", [])
              if tc["action"] == "get_order"]
        assert all(s != "success" for s in s1)
        assert all(s != "success" for s in s2)


# ===========================================================================
# Session validation
# ===========================================================================

class TestSessionValidation:

    def test_unknown_session_returns_safe_message(self, client):
        result = _post(client, "Where is my order?", "totally-unknown-session-xyz999")
        assert result["status"] == 200
        resp = result["body"]["response"].lower()
        assert "session" in resp or "log in" in resp or "verify" in resp

    def test_empty_session_id_returns_422(self, client):
        resp = client.post("/query", json={"message": "Hello", "session_id": ""})
        assert resp.status_code == 422

    def test_missing_session_id_returns_422(self, client):
        resp = client.post("/query", json={"message": "Hello"})
        assert resp.status_code == 422

    def test_missing_message_returns_422(self, client):
        resp = client.post("/query", json={"session_id": "sess-cust001"})
        assert resp.status_code == 422


# ===========================================================================
# Session ID injection
# ===========================================================================

class TestSessionIdInjection:

    @pytest.mark.parametrize("bad_session_id", [
        "sess'; DROP TABLE customers; --",
        "../../etc/passwd",
        "<script>alert(1)</script>",
        "sess cust001",
        "sess/cust001",
        'sess"cust001',
        "sess'cust001",
    ])
    def test_injection_session_id_returns_422(self, client, bad_session_id):
        resp = client.post("/query", json={"message": "Hello", "session_id": bad_session_id})
        assert resp.status_code == 422, (
            f"Expected 422 for session_id={bad_session_id!r}, got {resp.status_code}"
        )


# ===========================================================================
# Direct tool ownership tests
# ===========================================================================

class TestToolOwnershipDirect:

    def test_oms_get_order_rejects_wrong_customer(self, patched_env):
        import asyncio
        from tools.oms_tool import OmsTool
        from agent.graph import OwnershipError, _assert_ownership
        tool  = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(tool.get_order("ORD-99001"))
        assert order["customer_id"] == "CUST-002"
        with pytest.raises(OwnershipError):
            _assert_ownership("CUST-002", "CUST-001", "ORD-99001")

    def test_oms_get_order_allows_correct_customer(self, patched_env):
        import asyncio
        from tools.oms_tool import OmsTool
        from agent.graph import _assert_ownership
        tool  = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(tool.get_order("ORD-78321"))
        assert order["customer_id"] == "CUST-001"
        _assert_ownership("CUST-001", "CUST-001", "ORD-78321")

    def test_ownership_error_message_does_not_reveal_owner(self):
        from agent.graph import OwnershipError, _assert_ownership
        with pytest.raises(OwnershipError) as exc_info:
            _assert_ownership("CUST-002", "CUST-001", "ORD-99001")
        assert "CUST-002" not in str(exc_info.value)
