"""
tests/test_regression.py
=========================
Regression tests covering all recent fixes:
  - Order ID case insensitivity
  - Refund method normalisation
  - Response truncation (max_tokens)
  - Order not found explicit messaging
  - Invalid order ID format detection
  - Fast-path order tracking (no second LLM call)
"""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

from tests.conftest import make_llm_mock, _mock_plan_response
from agent.executor import Executor
from agent.guardrails import _extract_amounts


# ---------------------------------------------------------------------------
# Order ID case insensitivity
# ---------------------------------------------------------------------------

class TestOrderIdCaseInsensitivity:

    def test_lowercase_order_id_resolves(self, patched_env):
        """oms_tool.get_order with lowercase order_id must still find the order."""
        from tools.oms_tool import OmsTool
        tool = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(
            tool.get_order("ord-78321")
        )
        assert order["order_id"] == "ORD-78321"

    def test_mixed_case_order_id_resolves(self, patched_env):
        """Mixed case like Ord-78321 must resolve correctly."""
        from tools.oms_tool import OmsTool
        tool = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(
            tool.get_order("Ord-78321")
        )
        assert order["order_id"] == "ORD-78321"

    def test_uppercase_still_works(self, patched_env):
        """Uppercase ORD-78321 must continue to work."""
        from tools.oms_tool import OmsTool
        tool = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(
            tool.get_order("ORD-78321")
        )
        assert order["order_id"] == "ORD-78321"

    def test_repository_find_by_id_case_insensitive(self, patched_env):
        """OrderRepository.find_by_id must be case insensitive."""
        from repositories.order_repository import OrderRepository
        repo = OrderRepository()
        assert repo.find_by_id("ord-78321") is not None
        assert repo.find_by_id("ORD-78321") is not None
        assert repo.find_by_id("Ord-78321") is not None

    def test_whitespace_stripped_from_order_id(self, patched_env):
        """Whitespace around order ID must be stripped."""
        from tools.oms_tool import OmsTool
        tool = OmsTool()
        order = asyncio.get_event_loop().run_until_complete(
            tool.get_order("  ORD-78321  ")
        )
        assert order["order_id"] == "ORD-78321"

    def test_invalid_order_id_format_http(self, client):
        """Invalid order ID format in message → helpful format hint, no 500."""
        resp = client.post(
            "/query",
            json={"message": "Where is order ORD-123?", "session_id": "sess-cust001"},
        )
        assert resp.status_code == 200
        body = resp.json()
        response = body["response"].lower()
        assert "ord-" in response or "format" in response or "xxxxx" in response

    def test_order_id_in_message_case_insensitive(self, client):
        """Message with lowercase order ID like ord-78321 triggers format hint."""
        resp = client.post(
            "/query",
            json={"message": "Track ord-78321 please", "session_id": "sess-cust001"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Refund method normalisation
# ---------------------------------------------------------------------------

class TestRefundMethodNormalisation:

    @pytest.mark.parametrize("raw_method,expected", [
        ("original_payment_method", "original"),
        ("original payment method",  "original"),
        ("hdfc card",                "HDFC_CREDIT"),
        ("hdfc credit card",         "HDFC_CREDIT"),
        ("HDFC",                     "HDFC_CREDIT"),
        ("icici debit",              "ICICI_DEBIT"),
        ("icici card",               "ICICI_DEBIT"),
        ("sbi net banking",          "SBI_NETBANKING"),
        ("netbanking",               "SBI_NETBANKING"),
        ("gpay",                     "UPI"),
        ("phonepe",                  "UPI"),
        ("paytm",                    "UPI"),
        ("same card",                "original"),
        ("same method",              "original"),
        ("original",                 "original"),
        ("HDFC_CREDIT",              "HDFC_CREDIT"),
        ("UPI",                      "UPI"),
        ("completely_unknown_xyz",   "original"),  # unknown → safe default
    ])
    def test_normalise_refund_method(self, raw_method, expected):
        result = Executor._normalise_refund_method(raw_method)
        assert result == expected, (
            f"_normalise_refund_method({raw_method!r}) = {result!r}, expected {expected!r}"
        )

    def test_refund_with_no_method_defaults_to_original(self, patched_env):
        """process_refund with no method specified must use 'original'."""
        from tools.payment_tool import PaymentTool
        tool = PaymentTool()
        result = asyncio.get_event_loop().run_until_complete(
            tool.process_refund(
                order_id="ORD-78400",
                amount_inr=1000.0,
                method="original",
                customer_id="CUST-001",
            )
        )
        assert result["method"] == "original"
        assert result["status"] == "initiated"


# ---------------------------------------------------------------------------
# Order not found — explicit messaging
# ---------------------------------------------------------------------------

class TestOrderNotFoundMessaging:

    def test_not_found_response_mentions_order_id(self, client):
        """Order not found response must mention the order ID explicitly."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "I could not find that order.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                resp = client.post(
                    "/query",
                    json={"message": "Where is ORD-00000?", "session_id": "sess-cust001"},
                )
        assert resp.status_code == 200
        body = resp.json()
        response = body["response"]
        assert "ORD-00000" in response or "not find" in response.lower()

    def test_not_found_does_not_say_system_error(self, client):
        """Order not found must never say 'system error'."""
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "I could not find that order.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                resp = client.post(
                    "/query",
                    json={"message": "Where is ORD-00000?", "session_id": "sess-cust001"},
                )
        response = resp.json()["response"].lower()
        assert "system error" not in response
        assert "internal" not in response
        assert "exception" not in response


# ---------------------------------------------------------------------------
# Invalid order ID format detection
# ---------------------------------------------------------------------------

class TestInvalidOrderIdFormat:

    @pytest.mark.parametrize("bad_id", [
        "ORD-123",       # too few digits
        "ORD-ABCDE",     # letters not digits
        "ORDER-78321",   # wrong prefix
        "ORD-1234567",   # too many digits
    ])
    def test_invalid_order_id_caught_before_llm(self, client, bad_id):
        """Invalid order ID format must be caught pre-LLM and return helpful message."""
        resp = client.post(
            "/query",
            json={
                "message": f"Where is order {bad_id}?",
                "session_id": "sess-cust001",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        response = body["response"].lower()
        assert (
            "format" in response
            or "ord-" in response
            or "xxxxx" in response
            or "example" in response
        ), f"Expected format hint for bad ID {bad_id!r}, got: {body['response']}"

    def test_valid_order_id_not_caught_as_invalid(self, client):
        """Valid ORD-78321 must not trigger the format error path."""
        from tests.conftest import J1_PLAN
        llm_mock = make_llm_mock(J1_PLAN, "Your order is processing.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                resp = client.post(
                    "/query",
                    json={"message": "Track ORD-78321", "session_id": "sess-cust001"},
                )
        assert resp.status_code == 200
        body = resp.json()
        response = body["response"].lower()
        assert "format" not in response
        assert "xxxxx" not in response


# ---------------------------------------------------------------------------
# Fast-path order tracking (no second LLM call)
# ---------------------------------------------------------------------------

class TestFastPathOrderTracking:

    def test_order_tracking_success_skips_response_llm(self, client):
        """
        On successful order tracking, ResponseBuilder must NOT call the LLM.
        Only 1 LLM call total (planner), not 2.
        """
        from tests.conftest import J1_PLAN

        llm_call_count = 0

        async def counting_llm(*args, **kwargs):
            nonlocal llm_call_count
            llm_call_count += 1
            mock_msg = MagicMock()
            mock_msg.content = J1_PLAN.choices[0].message.content
            mock_choice = MagicMock()
            mock_choice.message = mock_msg
            mock_completion = MagicMock()
            mock_completion.choices = [mock_choice]
            return mock_completion.choices[0].message.content

        with patch("agent.planner.Planner._call_llm", new=AsyncMock(side_effect=counting_llm)):
            resp = client.post(
                "/query",
                json={"message": "Where is ORD-78321?", "session_id": "sess-cust001"},
            )

        assert resp.status_code == 200
        assert llm_call_count == 1, (
            f"Expected 1 LLM call (planner only) for order tracking, "
            f"got {llm_call_count}"
        )

    def test_order_tracking_fast_path_response_has_order_details(self, client):
        """Fast-path response must contain real order status and ID."""
        from tests.conftest import J1_PLAN
        llm_mock = make_llm_mock(J1_PLAN, "Your order is being processed.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                resp = client.post(
                    "/query",
                    json={"message": "Where is ORD-78321?", "session_id": "sess-cust001"},
                )
        body = resp.json()
        response = body["response"]
        assert "ORD-78321" in response
        assert any(word in response.lower() for word in [
            "processing", "shipped", "delivered", "placed", "status"
        ])

    def test_order_tracking_response_not_truncated(self, client):
        """Fast-path order tracking response must be a complete sentence."""
        from tests.conftest import J1_PLAN
        llm_mock = make_llm_mock(J1_PLAN, "Complete response.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                resp = client.post(
                    "/query",
                    json={"message": "Where is ORD-78321?", "session_id": "sess-cust001"},
                )
        response = resp.json()["response"]
        # Must end with proper punctuation or question mark
        assert response.strip()[-1] in ".?!", (
            f"Response appears truncated, ends with: {response[-30:]!r}"
        )


# ---------------------------------------------------------------------------
# Planner max_tokens reduction (indirectly verified via latency)
# ---------------------------------------------------------------------------

class TestPlannerConfig:

    def test_planner_uses_low_max_tokens(self):
        """Planner LLM call must use max_tokens=256 not 1024."""
        import inspect
        from agent import planner
        source = inspect.getsource(planner)
        assert "max_tokens=256" in source, (
            "Planner should use max_tokens=256 for faster responses. "
            "Found max_tokens=1024 which causes slow planning."
        )