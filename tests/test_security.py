"""
tests/test_security.py
=======================
Security and ownership enforcement tests.
"""

import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from tests.conftest import _mock_plan_response, make_llm_mock


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


def _run(client, message, plan, session_id="sess-cust001"):
    mock = make_llm_mock(plan, "Done.")
    with patch("agent.planner.Planner._call_llm", new=mock):
        with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
            return _post(client, message, session_id)


def _statuses(result): return [tc["status"] for tc in result["body"].get("trace", {}).get("tool_calls", [])]
def _actions(result):  return [tc["action"] for tc in result["body"].get("trace", {}).get("tool_calls", [])]


# ===========================================================================
# Cross-customer access denial
# ===========================================================================

class TestCrossCustomerAccess:

    def test_cust001_cannot_access_cust002_order(self, client):
        plan   = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-99001"}, "depends_on": []}])
        result = _run(client, "Where is order ORD-99001?", plan)
        assert result["status"] == 200
        assert "ownership_denied" in _statuses(result)

    def test_cust001_cannot_cancel_cust002_item(self, client):
        plan   = _mock_plan_response("partial_cancellation",
            [{"action": "cancel_item", "params": {"order_id": "ORD-99001", "line_id": 1}, "depends_on": []}])
        result = _run(client, "Cancel item 1 from ORD-99001.", plan)
        assert result["status"] == 200
        assert "ownership_denied" in _statuses(result)

    def test_cust001_cannot_refund_cust002_order(self, client):
        plan   = _mock_plan_response("refund_request",
            [{"action": "process_refund",
              "params": {"order_id": "ORD-99001", "amount_inr": 1000.0, "method": "HDFC_CREDIT"},
              "depends_on": []}])
        result = _run(client, "Refund ORD-99001.", plan)
        assert result["status"] == 200
        assert "ownership_denied" in _statuses(result)

    def test_cust001_cannot_update_address_of_cust002_order(self, client):
        plan   = _mock_plan_response("address_update",
            [{"action": "update_address",
              "params": {"order_id": "ORD-99001", "address_label": "home"},
              "depends_on": []}])
        result = _run(client, "Update address for ORD-99001.", plan)
        assert result["status"] == 200
        assert "ownership_denied" in _statuses(result)

    def test_ownership_error_message_is_vague(self, client):
        plan   = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-99001"}, "depends_on": []}])
        result = _run(client, "Track ORD-99001.", plan)
        resp   = result["body"].get("response", "").lower()
        assert "cust-002"         not in resp
        assert "another customer" not in resp
        assert "belongs to"       not in resp

    def test_nonexistent_order_same_response_as_other_customer_order(self, client):
        plan1  = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}])
        plan2  = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-99001"}, "depends_on": []}])
        r1     = _run(client, "Track ORD-00000.", plan1)
        r2     = _run(client, "Track ORD-99001.", plan2)
        # Neither should have a success tool call for get_order
        s1 = [tc["status"] for tc in r1["body"].get("trace", {}).get("tool_calls", [])
              if tc["action"] == "get_order"]
        s2 = [tc["status"] for tc in r2["body"].get("trace", {}).get("tool_calls", [])
              if tc["action"] in ("get_order",)]
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
        from agent.executor import Executor, OwnershipError
        tool  = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(tool.get_order("ORD-99001"))
        assert order["customer_id"] == "CUST-002"
        with pytest.raises(OwnershipError):
            Executor._assert_ownership("CUST-002", "CUST-001", "ORD-99001")

    def test_oms_get_order_allows_correct_customer(self, patched_env):
        import asyncio
        from tools.oms_tool import OmsTool
        from agent.executor import Executor
        tool  = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(tool.get_order("ORD-78321"))
        assert order["customer_id"] == "CUST-001"
        Executor._assert_ownership("CUST-001", "CUST-001", "ORD-78321")

    def test_ownership_error_message_does_not_reveal_owner(self):
        from agent.executor import Executor, OwnershipError
        with pytest.raises(OwnershipError) as exc_info:
            Executor._assert_ownership("CUST-002", "CUST-001", "ORD-99001")
        assert "CUST-002" not in str(exc_info.value)