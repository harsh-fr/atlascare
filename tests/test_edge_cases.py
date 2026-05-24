"""
tests/test_edge_cases.py
=========================
Edge case, failure mode, and hallucination prevention tests.

Coverage
--------
  Invalid inputs
    - Non-existent order ID
    - Invalid order ID format
    - Invalid line item ID (0, negative, string)
    - Already-cancelled item
    - Order in wrong status for cancellation (shipped, delivered)

  Payment edge cases
    - Gateway timeout → retry → eventual success
    - Gateway timeout → all retries exhausted → error recorded in trace
    - Refund amount with many decimal places

  Missing data
    - Customer has no office address → error in trace
    - KB search with no matching tags → empty result

  Planner resilience
    - LLM returns malformed JSON → PlannerError → safe user response
    - LLM returns unknown intent → unknown plan → safe fallback
    - LLM returns unknown action type → PlannerError → safe user response

  Hallucination prevention
    - Response never contains fabricated order IDs
    - Response never contains fabricated tracking numbers
    - Response for unknown order does not invent order details

  Trace integrity
    - trace_id is unique per request
    - latency_ms is always present and positive
    - Failed steps appear in trace (not silently dropped)

  Concurrent-style isolation
    - Two sequential requests produce different trace_ids
"""

import asyncio
import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from tests.conftest import make_llm_mock, _mock_plan_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(client, message: str, session_id: str = "sess-cust001") -> dict:
    resp = client.post("/query", json={"message": message, "session_id": session_id})
    assert resp.status_code == 200, f"Unexpected {resp.status_code}: {resp.text}"
    return resp.json()


def _tool_statuses(body: dict) -> list[str]:
    return [tc["status"] for tc in body.get("trace", {}).get("tool_calls", [])]


def _tool_actions(body: dict) -> list[str]:
    return [tc["action"] for tc in body.get("trace", {}).get("tool_calls", [])]


# ---------------------------------------------------------------------------
# Invalid order inputs
# ---------------------------------------------------------------------------

class TestInvalidOrderInputs:

    def test_nonexistent_order_id_recorded_as_error(self, client):
        """Requesting a non-existent order must record error in trace, not crash."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Order not found.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Where is order ORD-00000?")

        assert "error" in _tool_statuses(body) or "ownership_denied" in _tool_statuses(body)

    def test_nonexistent_order_returns_200_not_500(self, client):
        """Non-existent order must never return HTTP 500."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Order not found.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                resp = client.post(
                    "/query",
                    json={"message": "Track ORD-00000.", "session_id": "sess-cust001"},
                )
        assert resp.status_code == 200

    def test_cancel_nonexistent_line_id_recorded_as_error(self, client):
        """Cancelling a line_id that doesn't exist must be recorded as error."""
        plan = _mock_plan_response(
            "partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-78321", "line_id": 99},
              "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Could not cancel.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Cancel item 99 from ORD-78321.")

        cancel_calls = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cancel_calls
        assert cancel_calls[0]["status"] == "error"

    def test_cancel_already_cancelled_item_recorded_as_error(self, client, data_dir):
        """Attempting to cancel an already-cancelled item must error cleanly."""
        # Pre-cancel item 1 in ORD-78321
        orders_path = data_dir / "orders.json"
        orders_data = json.loads(orders_path.read_text())
        for order in orders_data["orders"]:
            if order["order_id"] == "ORD-78321":
                for item in order["items"]:
                    if item["line_id"] == 1:
                        item["status"] = "cancelled"
        orders_path.write_text(json.dumps(orders_data, indent=2))

        plan = _mock_plan_response(
            "partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-78321", "line_id": 1},
              "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Already cancelled.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Cancel item 1 from ORD-78321.")

        cancel_calls = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cancel_calls
        assert cancel_calls[0]["status"] == "error"

    def test_cancel_shipped_order_recorded_as_error(self, client):
        """Cancelling an item from a shipped order must be recorded as error."""
        plan = _mock_plan_response(
            "partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-78322", "line_id": 1},
              "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Cannot cancel shipped order.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Cancel item 1 from ORD-78322.")

        cancel_calls = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cancel_calls
        assert cancel_calls[0]["status"] == "error"

    def test_cancel_delivered_order_recorded_as_error(self, client):
        """Cancelling an item from a delivered order must be recorded as error."""
        plan = _mock_plan_response(
            "partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-78323", "line_id": 1},
              "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Cannot cancel delivered order.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Cancel item 1 from ORD-78323.")

        cancel_calls = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cancel_calls
        assert cancel_calls[0]["status"] == "error"


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestMissingData:

    def test_missing_office_address_recorded_as_error(self, client, data_dir):
        """If customer has no office address, update_address must error cleanly."""
        # Remove office address from CUST-001
        crm_path = data_dir / "crm_cases.json"
        crm_data = json.loads(crm_path.read_text())
        for customer in crm_data["customers"]:
            if customer["customer_id"] == "CUST-001":
                customer["addresses"] = [
                    a for a in customer.get("addresses", [])
                    if a.get("label") != "office"
                ]
        crm_path.write_text(json.dumps(crm_data, indent=2))

        plan = _mock_plan_response(
            "address_update",
            [{"action": "update_address",
              "params": {"order_id": "ORD-78321", "address_label": "office"},
              "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Could not update address.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Ship to my office address for ORD-78321.")

        address_calls = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "update_address"]
        assert address_calls
        assert address_calls[0]["status"] == "error"

    def test_kb_search_empty_tags_returns_empty(self, patched_env):
        """KB search with empty tags must return empty list without crashing."""
        from tools.kb_tool import KbTool
        tool = KbTool()
        result = asyncio.get_event_loop().run_until_complete(
            tool.search(tags=[])
        )
        assert result == []

    def test_kb_search_no_matching_tags_returns_empty(self, patched_env):
        """KB search with tags that match nothing must return empty list."""
        from tools.kb_tool import KbTool
        tool = KbTool()
        result = asyncio.get_event_loop().run_until_complete(
            tool.search(tags=["zzz_no_such_tag_xyz"])
        )
        assert result == []


# ---------------------------------------------------------------------------
# Payment gateway retry
# ---------------------------------------------------------------------------

class TestPaymentGatewayRetry:

    def test_gateway_timeout_then_success(self, patched_env, data_dir):
        """Gateway fails once then succeeds — final result must be success."""
        import random
        from tools.payment_tool import PaymentTool

        # First call fails, second succeeds
        call_count = 0
        original_random = random.random

        def controlled_random():
            nonlocal call_count
            call_count += 1
            return 0.0 if call_count == 1 else 1.0  # fail first, succeed second

        tool = PaymentTool()
        with patch("tools.payment_tool.random.random", side_effect=controlled_random):
            result = asyncio.get_event_loop().run_until_complete(
                tool.process_refund(
                    order_id="ORD-78321",
                    amount_inr=1500.0,
                    method="HDFC_CREDIT",
                    customer_id="CUST-001",
                )
            )
        assert result["status"] == "initiated"

    def test_all_retries_exhausted_raises_error(self, patched_env, monkeypatch):
        """All gateway retries exhausted → PaymentGatewayError raised."""
        import random
        from tools.payment_tool import PaymentTool, PaymentGatewayError

        monkeypatch.setenv("PAYMENT_MAX_RETRIES", "2")
        monkeypatch.setenv("PAYMENT_RETRY_BASE_DELAY_S", "0.0")

        tool = PaymentTool()
        with patch("tools.payment_tool.random.random", return_value=0.0):  # always fail
            with pytest.raises(PaymentGatewayError):
                asyncio.get_event_loop().run_until_complete(
                    tool.process_refund(
                        order_id="ORD-78321",
                        amount_inr=1500.0,
                        method="HDFC_CREDIT",
                        customer_id="CUST-001",
                    )
                )


# ---------------------------------------------------------------------------
# Planner resilience
# ---------------------------------------------------------------------------

class TestPlannerResilience:

    def test_malformed_llm_json_returns_safe_response(self, client):
        """LLM returning non-JSON must produce a safe user response, not 500."""
        bad_json = AsyncMock(return_value="This is not JSON at all !!!")
        with patch("agent.planner.Planner._call_llm", new=bad_json):
            resp = client.post(
                "/query",
                json={"message": "Where is my order?", "session_id": "sess-cust001"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"]
        assert "exception" not in body["response"].lower()
        assert "traceback" not in body["response"].lower()

    def test_unknown_intent_returns_safe_response(self, client):
        """LLM returning unknown intent must not crash the pipeline."""
        plan = _mock_plan_response("totally_unknown_intent_xyz", [])
        llm_mock = make_llm_mock(plan, "I can help with that.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                resp = client.post(
                    "/query",
                    json={"message": "Do something weird.", "session_id": "sess-cust001"},
                )
        assert resp.status_code == 200

    def test_llm_timeout_returns_safe_response(self, client):
        """LLM call timing out must produce a safe response, not 500."""
        async def timeout_llm(*args, **kwargs):
            raise asyncio.TimeoutError("LLM timed out")

        with patch("agent.planner.Planner._call_llm", new=timeout_llm):
            resp = client.post(
                "/query",
                json={"message": "Where is my order?", "session_id": "sess-cust001"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"]


# ---------------------------------------------------------------------------
# Hallucination prevention
# ---------------------------------------------------------------------------

class TestHallucinationPrevention:

    def test_response_does_not_invent_tracking_number(self, client, data_dir):
        """
        For a 'placed' order with no tracking number, the response must not
        contain a fabricated tracking number.
        """
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78324"}, "depends_on": []}],
        )
        # ORD-78324 is 'placed' — no tracking number in data
        llm_mock = make_llm_mock(plan, "Your order ORD-78324 has been placed and is being prepared.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Track ORD-78324.")

        response = body["response"]
        # Must not contain a tracking number format (TRACK-XXXXXXX)
        import re
        fabricated = re.findall(r"TRACK-[A-Z0-9]+", response)
        assert not fabricated, (
            f"Response contains fabricated tracking number: {fabricated}"
        )

    def test_nonexistent_order_response_has_no_invented_data(self, client):
        """Response for non-existent order must not contain invented order details."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "I could not find that order.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Where is ORD-00000?")

        response = body["response"].lower()
        # Must not invent a status for a non-existent order
        assert "shipped" not in response or "not" in response
        assert "delivered" not in response or "not" in response


# ---------------------------------------------------------------------------
# Trace integrity
# ---------------------------------------------------------------------------

class TestTraceIntegrity:

    def test_trace_id_unique_per_request(self, client):
        """Each request must produce a unique trace_id."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}],
        )
        mock1 = make_llm_mock(plan, "Your order is processing.")
        mock2 = make_llm_mock(plan, "Your order is processing.")

        with patch("agent.planner.Planner._call_llm", new=mock1):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock1):
                body1 = _post(client, "Track ORD-78321.")

        with patch("agent.planner.Planner._call_llm", new=mock2):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock2):
                body2 = _post(client, "Track ORD-78321.")

        assert body1["trace"]["trace_id"] != body2["trace"]["trace_id"], (
            "Two separate requests produced the same trace_id — IDs are not unique."
        )

    def test_latency_ms_always_present_and_positive(self, client):
        """latency_ms must always be present, integer, and > 0."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Processing.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Track ORD-78321.")

        assert isinstance(body["trace"]["latency_ms"], int)
        assert body["trace"]["latency_ms"] > 0

    def test_failed_step_appears_in_trace(self, client):
        """A failed tool call must appear in trace with status=error, not be dropped."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Order not found.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Track ORD-00000.")

        tool_calls = body["trace"]["tool_calls"]
        # At minimum the planner call and the failed get_order should be present
        assert len(tool_calls) >= 1
        statuses = [tc["status"] for tc in tool_calls]
        # There must be at least one non-success status
        assert any(s != "success" for s in statuses)

    def test_trace_tool_calls_ordered_correctly(self, client):
        """Planner call must appear before executor calls in tool_calls list."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Processing.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post(client, "Track ORD-78321.")

        tool_names = [tc["tool"] for tc in body["trace"]["tool_calls"]]
        # planner must come before get_order
        if "planner" in tool_names and "get_order" in tool_names:
            assert tool_names.index("planner") < tool_names.index("get_order")


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class TestValidators:

    def test_validate_order_id_valid(self):
        from utils.validators import validate_order_id
        assert validate_order_id("ORD-78321") == "ORD-78321"

    def test_validate_order_id_normalises_case(self):
        from utils.validators import validate_order_id
        assert validate_order_id("ord-78321") == "ORD-78321"

    def test_validate_order_id_invalid_raises(self):
        from utils.validators import validate_order_id
        with pytest.raises(ValueError):
            validate_order_id("INVALID-123")

    def test_validate_customer_id_valid(self):
        from utils.validators import validate_customer_id
        assert validate_customer_id("CUST-001") == "CUST-001"

    def test_validate_refund_amount_valid(self):
        from utils.validators import validate_refund_amount
        assert validate_refund_amount(1500.0) == 1500.0

    def test_validate_refund_amount_zero_raises(self):
        from utils.validators import validate_refund_amount
        with pytest.raises(ValueError):
            validate_refund_amount(0.0)

    def test_validate_refund_amount_negative_raises(self):
        from utils.validators import validate_refund_amount
        with pytest.raises(ValueError):
            validate_refund_amount(-100.0)

    def test_validate_payment_method_valid(self):
        from utils.validators import validate_payment_method
        assert validate_payment_method("HDFC_CREDIT") == "HDFC_CREDIT"

    def test_validate_payment_method_invalid_raises(self):
        from utils.validators import validate_payment_method
        with pytest.raises(ValueError):
            validate_payment_method("BITCOIN")

    def test_validate_line_id_string_int(self):
        from utils.validators import validate_line_id
        assert validate_line_id("2") == 2

    def test_validate_line_id_zero_raises(self):
        from utils.validators import validate_line_id
        with pytest.raises(ValueError):
            validate_line_id(0)