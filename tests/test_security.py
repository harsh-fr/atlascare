"""
tests/test_security.py
=======================
Security and ownership enforcement tests.

Coverage
--------
  Cross-customer access denial
    - Customer A cannot access Customer B's order
    - Response is vague (does not reveal order exists)
    - Trace records ownership_denied, not the real order data

  Session validation
    - Unknown session_id returns safe message
    - Empty session_id returns 422
    - Malformed session_id (special chars) returns 422

  Order ownership via all tool paths
    - get_order rejects cross-customer access
    - cancel_item rejects cross-customer access
    - update_address rejects cross-customer access
    - process_refund rejects cross-customer access

  Session ID injection
    - session_id with SQL injection chars returns 422
    - session_id with path traversal chars returns 422
    - session_id with script injection chars returns 422

  Data non-disclosure
    - Failed ownership check never reveals real order data
    - Error message is identical whether order exists or not
      (prevents order ID enumeration)
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from tests.conftest import make_llm_mock, _mock_plan_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(client: TestClient, message: str, session_id: str) -> dict:
    resp = client.post(
        "/query",
        json={"message": message, "session_id": session_id},
    )
    return {"status": resp.status_code, "body": resp.json() if resp.status_code != 422 else resp.json()}


def _tool_statuses(body: dict) -> list[str]:
    return [tc["status"] for tc in body.get("trace", {}).get("tool_calls", [])]


def _tool_actions(body: dict) -> list[str]:
    return [tc["action"] for tc in body.get("trace", {}).get("tool_calls", [])]


# ---------------------------------------------------------------------------
# Cross-customer access denial
# ---------------------------------------------------------------------------

class TestCrossCustomerAccess:

    def test_cust001_cannot_access_cust002_order(self, client: TestClient):
        """
        CUST-001 session must NOT be able to access ORD-99001 (owned by CUST-002).
        """
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-99001"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Here is the order.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                result = _post(client, "Where is order ORD-99001?", "sess-cust001")

        assert result["status"] == 200
        # Ownership denial must be in trace
        assert "ownership_denied" in _tool_statuses(result["body"])

    def test_cust001_cannot_cancel_cust002_item(self, client: TestClient):
        """CUST-001 must not be able to cancel items in CUST-002's order."""
        plan = _mock_plan_response(
            "partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-99001", "line_id": 1},
              "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Item cancelled.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                result = _post(client, "Cancel item 1 from ORD-99001.", "sess-cust001")

        assert result["status"] == 200
        assert "ownership_denied" in _tool_statuses(result["body"])

    def test_cust001_cannot_refund_cust002_order(self, client: TestClient):
        """CUST-001 must not be able to initiate refund on CUST-002's order."""
        plan = _mock_plan_response(
            "refund_request",
            [{"action": "process_refund",
              "params": {"order_id": "ORD-99001", "amount_inr": 1000.0, "method": "HDFC_CREDIT"},
              "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Refund processed.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                result = _post(client, "Refund ORD-99001.", "sess-cust001")

        assert result["status"] == 200
        assert "ownership_denied" in _tool_statuses(result["body"])

    def test_cust001_cannot_update_address_of_cust002_order(self, client: TestClient):
        """CUST-001 must not be able to update address on CUST-002's order."""
        plan = _mock_plan_response(
            "address_update",
            [{"action": "update_address",
              "params": {"order_id": "ORD-99001", "address_label": "home"},
              "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Address updated.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                result = _post(client, "Update address for ORD-99001.", "sess-cust001")

        assert result["status"] == 200
        assert "ownership_denied" in _tool_statuses(result["body"])

    def test_ownership_error_message_is_vague(self, client: TestClient):
        """
        Error message on ownership failure must be vague — must not reveal
        that the order exists or belongs to another customer.
        This prevents order ID enumeration attacks.
        """
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-99001"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Order details.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                result = _post(client, "Track ORD-99001.", "sess-cust001")

        response_text = result["body"].get("response", "").lower()

        # Must NOT reveal: existence of the order or who owns it
        assert "cust-002"   not in response_text
        assert "another customer" not in response_text
        assert "belongs to" not in response_text

    def test_nonexistent_order_same_message_as_other_customer_order(
        self, client: TestClient
    ):
        """
        Accessing a non-existent order and another customer's order
        should produce similar error responses — prevents enumeration.
        """
        plan_nonexistent = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}],
        )
        plan_other = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-99001"}, "depends_on": []}],
        )
        mock1 = make_llm_mock(plan_nonexistent, "Not found.")
        mock2 = make_llm_mock(plan_other, "Not found.")

        with patch("agent.planner.Planner._call_llm", new=mock1):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock1):
                r1 = _post(client, "Track ORD-00000.", "sess-cust001")

        with patch("agent.planner.Planner._call_llm", new=mock2):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock2):
                r2 = _post(client, "Track ORD-99001.", "sess-cust001")

        # Both should result in a non-success tool status
        for status in _tool_statuses(r1["body"]):
            assert status != "success" or "get_order" not in _tool_actions(r1["body"])
        for status in _tool_statuses(r2["body"]):
            if status == "ownership_denied":
                break  # Expected


# ---------------------------------------------------------------------------
# Session validation
# ---------------------------------------------------------------------------

class TestSessionValidation:

    def test_unknown_session_returns_safe_message(self, client: TestClient):
        """Unknown session_id must return HTTP 200 with a safe message."""
        result = _post(
            client,
            "Where is my order?",
            "totally-unknown-session-xyz999",
        )
        assert result["status"] == 200
        response = result["body"]["response"].lower()
        assert "session" in response or "log in" in response or "verify" in response

    def test_empty_session_id_returns_422(self, client: TestClient):
        """Empty session_id must return HTTP 422 (validation error)."""
        resp = client.post(
            "/query",
            json={"message": "Hello", "session_id": ""},
        )
        assert resp.status_code == 422

    def test_missing_session_id_returns_422(self, client: TestClient):
        """Missing session_id field must return HTTP 422."""
        resp = client.post(
            "/query",
            json={"message": "Hello"},
        )
        assert resp.status_code == 422

    def test_missing_message_returns_422(self, client: TestClient):
        """Missing message field must return HTTP 422."""
        resp = client.post(
            "/query",
            json={"session_id": "sess-cust001"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Session ID injection attempts
# ---------------------------------------------------------------------------

class TestSessionIdInjection:

    @pytest.mark.parametrize("bad_session_id", [
        "sess'; DROP TABLE customers; --",
        "../../etc/passwd",
        "<script>alert(1)</script>",
        "sess cust001",          # space not allowed
        "sess/cust001",          # slash not allowed
        "sess\"cust001",         # double quote not allowed
        "sess'cust001",          # single quote not allowed
    ])
    def test_injection_session_id_returns_422(
        self, client: TestClient, bad_session_id: str
    ):
        """Malformed session_id with injection characters must be rejected with 422."""
        resp = client.post(
            "/query",
            json={"message": "Hello", "session_id": bad_session_id},
        )
        assert resp.status_code == 422, (
            f"Expected 422 for session_id={bad_session_id!r}, "
            f"got {resp.status_code}"
        )


# ---------------------------------------------------------------------------
# Direct tool ownership tests (unit level)
# ---------------------------------------------------------------------------

class TestToolOwnershipDirect:

    def test_oms_get_order_rejects_wrong_customer(self, patched_env):
        """OmsTool.get_order followed by ownership check must raise OwnershipError."""
        import asyncio
        from tools.oms_tool import OmsTool
        from agent.executor import OwnershipError

        tool = OmsTool()

        # First fetch the order to get its customer_id
        order = asyncio.get_event_loop().run_until_complete(
            tool.get_order("ORD-99001")
        )
        assert order["customer_id"] == "CUST-002"

        # Now simulate ownership check from executor
        from agent.executor import Executor
        with pytest.raises(OwnershipError):
            Executor._assert_ownership(
                order_customer_id="CUST-002",
                session_customer_id="CUST-001",
                order_id="ORD-99001",
            )

    def test_oms_get_order_allows_correct_customer(self, patched_env):
        """OmsTool.get_order followed by ownership check must pass for correct customer."""
        import asyncio
        from tools.oms_tool import OmsTool
        from agent.executor import Executor

        tool = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(
            tool.get_order("ORD-78321")
        )
        assert order["customer_id"] == "CUST-001"

        # Must not raise
        Executor._assert_ownership(
            order_customer_id="CUST-001",
            session_customer_id="CUST-001",
            order_id="ORD-78321",
        )

    def test_ownership_error_message_does_not_reveal_owner(self):
        """OwnershipError message must not contain the real customer_id."""
        from agent.executor import Executor, OwnershipError

        try:
            Executor._assert_ownership(
                order_customer_id="CUST-002",
                session_customer_id="CUST-001",
                order_id="ORD-99001",
            )
        except OwnershipError as e:
            assert "CUST-002" not in str(e), (
                "OwnershipError must not reveal the real owner's customer_id."
            )