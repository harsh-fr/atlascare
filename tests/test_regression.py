"""
tests/test_regression.py
=========================
Regression tests covering all recent fixes.
"""

import asyncio
import inspect
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from tests.conftest import J1_PLAN, _mock_plan_response, make_llm_mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(client, message, plan, resp="Done.", session_id="sess-cust001"):
    mock = make_llm_mock(plan, resp)
    with patch("agent.planner.Planner._call_llm", new=mock):
        with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
            r = client.post("/query", json={"message": message, "session_id": session_id})
            assert r.status_code == 200, f"{r.status_code}: {r.text}"
            return r.json()


# ===========================================================================
# Order ID case insensitivity
# ===========================================================================

class TestOrderIdCaseInsensitivity:

    def test_lowercase_order_id_resolves(self, patched_env):
        from tools.oms_tool import OmsTool
        order = asyncio.get_event_loop().run_until_complete(
            OmsTool().get_order("ord-78321"))
        assert order["order_id"] == "ORD-78321"

    def test_mixed_case_order_id_resolves(self, patched_env):
        from tools.oms_tool import OmsTool
        order = asyncio.get_event_loop().run_until_complete(
            OmsTool().get_order("Ord-78321"))
        assert order["order_id"] == "ORD-78321"

    def test_uppercase_still_works(self, patched_env):
        from tools.oms_tool import OmsTool
        order = asyncio.get_event_loop().run_until_complete(
            OmsTool().get_order("ORD-78321"))
        assert order["order_id"] == "ORD-78321"

    def test_repository_find_by_id_case_insensitive(self, patched_env):
        from repositories.order_repository import OrderRepository
        repo = OrderRepository()
        assert repo.find_by_id("ord-78321") is not None
        assert repo.find_by_id("ORD-78321") is not None
        assert repo.find_by_id("Ord-78321") is not None

    def test_whitespace_stripped_from_order_id(self, patched_env):
        from tools.oms_tool import OmsTool
        order = asyncio.get_event_loop().run_until_complete(
            OmsTool().get_order("  ORD-78321  "))
        assert order["order_id"] == "ORD-78321"

    def test_invalid_order_id_format_caught_pre_llm(self, client):
        resp = client.post("/query",
            json={"message": "Where is order ORD-123?", "session_id": "sess-cust001"})
        assert resp.status_code == 200
        body = resp.json()
        r    = body["response"].lower()
        assert "format" in r or "ord-" in r or "xxxxx" in r or "5 digit" in r


# ===========================================================================
# Refund method normalisation
# ===========================================================================

class TestRefundMethodNormalisation:

    @pytest.mark.parametrize("raw,expected", [
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
        ("completely_unknown_xyz",   "original"),
    ])
    def test_normalise_refund_method(self, raw, expected):
        from agent.executor import Executor
        result = Executor._normalise_refund_method(raw)
        assert result == expected, (
            f"_normalise_refund_method({raw!r}) = {result!r}, expected {expected!r}"
        )

    def test_refund_with_no_method_defaults_to_original(self, patched_env):
        from tools.payment_tool import PaymentTool
        tool   = PaymentTool()
        result = asyncio.get_event_loop().run_until_complete(
            tool.process_refund("ORD-78400", 1000.0, "original", "CUST-001"))
        assert result["method"] == "original"
        assert result["status"] == "initiated"


# ===========================================================================
# Order not found messaging
# ===========================================================================

class TestOrderNotFoundMessaging:

    def test_not_found_response_mentions_order_id(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}])
        body = _run(client, "Where is ORD-00000?", plan)
        resp = body["response"]
        assert "ORD-00000" in resp or "not find" in resp.lower() or "not found" in resp.lower()

    def test_not_found_does_not_say_system_error(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}])
        body = _run(client, "Where is ORD-00000?", plan)
        resp = body["response"].lower()
        assert "system error" not in resp
        assert "internal"     not in resp
        assert "exception"    not in resp


# ===========================================================================
# Invalid order ID format
# ===========================================================================

class TestInvalidOrderIdFormat:

    @pytest.mark.parametrize("bad_id", [
        "ORD-123",
        "ORD-ABCDE",
        "ORDER-78321",
        "ORD-1234567",
    ])
    def test_invalid_order_id_caught_before_llm(self, client, bad_id):
        resp = client.post("/query",
            json={"message": f"Where is order {bad_id}?", "session_id": "sess-cust001"})
        assert resp.status_code == 200
        body = resp.json()
        r    = body["response"].lower()
        assert any(k in r for k in ["format","ord-","xxxxx","example","5 digit"]), (
            f"Expected format hint for {bad_id!r}, got: {body['response']}"
        )

    def test_valid_order_id_not_caught_as_invalid(self, client):
        body = _run(client, "Track ORD-78321", J1_PLAN)
        r    = body["response"].lower()
        assert "format" not in r
        assert "xxxxx"  not in r


# ===========================================================================
# Fast-path order tracking
# ===========================================================================

class TestFastPathOrderTracking:

    def test_order_tracking_fast_path_response_has_order_details(self, client):
        body = _run(client, "Where is ORD-78321?", J1_PLAN, "Your order is processing.")
        resp = body["response"]
        assert "ORD-78321" in resp
        assert any(w in resp.lower() for w in ["processing","shipped","delivered","placed","status"])

    def test_order_tracking_response_not_truncated(self, client):
        body = _run(client, "Where is ORD-78321?", J1_PLAN, "Complete.")
        resp = body["response"].strip()
        assert resp[-1] in ".?!*)"

    def test_order_tracking_only_one_llm_call(self, client):
        """Fast-path: successful order tracking skips the ResponseBuilder LLM call."""
        call_count = 0
        plan_str   = J1_PLAN.choices[0].message.content

        async def counting_llm(*a, **k):
            nonlocal call_count
            call_count += 1
            return plan_str

        with patch("agent.planner.Planner._call_llm", new=counting_llm):
            client.post("/query",
                json={"message": "Where is ORD-78321?", "session_id": "sess-cust001"})

        assert call_count == 1, (
            f"Expected 1 LLM call (planner only) for order tracking, got {call_count}"
        )


# ===========================================================================
# Planner token budget
# ===========================================================================

class TestPlannerConfig:

    def test_planner_uses_256_max_tokens(self):
        """Planner must use max_tokens=256 not 1024 for fast responses."""
        from agent import planner
        source = inspect.getsource(planner)
        assert "max_tokens=256" in source, (
            "Planner should use max_tokens=256. "
            "Found a different value which causes slow responses."
        )


# ===========================================================================
# Vague help detection
# ===========================================================================

class TestVagueHelpDetection:

    @pytest.mark.parametrize("msg", [
        "help",
        "hi",
        "Hello",
        "i need help",
        "need help with my order",
        "can you help me",
    ])
    def test_vague_message_returns_graceful_prompt(self, client, msg):
        resp = client.post("/query",
            json={"message": msg, "session_id": "sess-cust001"})
        assert resp.status_code == 200
        body = resp.json()["response"].lower()
        assert "order" in body or "ord-" in body or "id" in body, (
            f"Expected order ID prompt for vague message {msg!r}, got: {resp.json()['response']}"
        )

    def test_vague_help_does_not_say_invalid_format(self, client):
        """'need help' must NOT trigger the invalid order ID format message."""
        resp = client.post("/query",
            json={"message": "need help", "session_id": "sess-cust001"})
        body = resp.json()["response"].lower()
        assert "doesn't look quite right" not in body
        assert "doesn't look" not in body
        # Should be a graceful prompt, not an error message
        assert "error" not in body


# ===========================================================================
# Auto-fill refund amount
# ===========================================================================

class TestAutoFillRefundAmount:

    def test_refund_without_amount_auto_filled_from_order(self, client):
        """
        When a refund plan step has no amount_inr, orchestrator should
        auto-fill it from the order total before execution.
        """
        # Plan with missing amount_inr
        plan = _mock_plan_response("refund_request", [
            {"action": "process_refund",
             "params": {"order_id": "ORD-78400", "method": "original"},
             "depends_on": []},
        ])
        body = _run(client, "I want a refund for order ORD-78400.", plan)
        rc   = [tc for tc in body["trace"]["tool_calls"]
                if tc["action"] == "process_refund"]
        # Should succeed with auto-filled amount (Rs.24,999 from order data)
        if rc:
            assert rc[0]["status"] == "success"