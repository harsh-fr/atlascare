"""
tests/test_edge_cases.py
=========================
Edge cases, failure modes, and hallucination prevention tests.
"""

import asyncio
import json
import re
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from tests.conftest import (
    make_tool_mock, make_multi_tool_mock, make_done_mock, make_text_mock,
    make_approved_mock, _make_order,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(client, message, groq_responses, session_id="sess-cust001"):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=groq_responses)
    with patch("agent.graph._groq_client", mock_client):
        resp_obj = client.post("/query", json={"message": message, "session_id": session_id})
        assert resp_obj.status_code == 200, f"{resp_obj.status_code}: {resp_obj.text}"
        return resp_obj.json()


def _post(client, message, session_id="sess-cust001"):
    resp = client.post("/query", json={"message": message, "session_id": session_id})
    assert resp.status_code == 200
    return resp.json()


def _statuses(body): return [tc["status"] for tc in body.get("trace", {}).get("tool_calls", [])]
def _actions(body):  return [tc["action"] for tc in body.get("trace", {}).get("tool_calls", [])]


# ===========================================================================
# Invalid order inputs
# ===========================================================================

class TestInvalidOrderInputs:

    def test_nonexistent_order_id_recorded_as_error(self, client):
        body = _run(client, "Where is order ORD-00000?", [
            make_tool_mock("get_order", {"order_id": "ORD-00000"}),
            make_text_mock("I couldn't find order ORD-00000."),
        ])
        s = _statuses(body)
        assert "error" in s or "ownership_denied" in s

    def test_nonexistent_order_returns_200_not_500(self, client):
        body = _run(client, "Track ORD-00000.", [
            make_tool_mock("get_order", {"order_id": "ORD-00000"}),
            make_text_mock("Order not found."),
        ])
        assert body  # 200 asserted inside _run

    def test_cancel_nonexistent_line_id_recorded_as_error(self, client):
        body = _run(client, "Cancel item 99 from ORD-78321.", [
            make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 99}),
            make_text_mock("That item doesn't exist."),
            make_approved_mock(),
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_cancel_already_cancelled_item_recorded_as_error(self, client, data_dir):
        import copy
        from repositories.order_repository import OrderRepository

        base_order      = _make_order("ORD-78321", "CUST-001", "processing")
        cancelled_order = copy.deepcopy(base_order)
        for i in cancelled_order["items"]:
            if i["line_id"] == 1:
                i["status"] = "cancelled"

        original_find = OrderRepository.find_by_id

        def patched_find(self_repo, order_id):
            if order_id.upper() == "ORD-78321":
                return copy.deepcopy(cancelled_order)
            return original_find(self_repo, order_id)

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 1}),
            make_text_mock("That item is already cancelled."),
            make_approved_mock(),
        ])
        with patch("agent.graph._groq_client", mock_client):
            with patch.object(OrderRepository, "find_by_id", patched_find):
                resp = client.post("/query",
                    json={"message": "Cancel item 1 from ORD-78321.",
                          "session_id": "sess-cust001"})

        assert resp.status_code == 200
        body = resp.json()
        cc   = [tc for tc in body["trace"]["tool_calls"]
                if tc["action"] == "cancel_item"]
        assert cc, "cancel_item not found in tool_calls"
        assert cc[0]["status"] == "error"

    def test_cancel_shipped_order_recorded_as_error(self, client):
        body = _run(client, "Cancel item 1 from ORD-78322.", [
            make_tool_mock("cancel_item", {"order_id": "ORD-78322", "line_id": 1}),
            make_text_mock("That order is shipped and cannot be cancelled."),
            make_approved_mock(),
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_cancel_delivered_order_recorded_as_error(self, client):
        body = _run(client, "Cancel item 1 from ORD-78323.", [
            make_tool_mock("cancel_item", {"order_id": "ORD-78323", "line_id": 1}),
            make_text_mock("That order is delivered and cannot be cancelled."),
            make_approved_mock(),
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_high_value_cancel_escalates_refund_to_specialist(self, client, data_dir):
        """Cancelling a >Rs.25,000 item must hand the refund to a specialist via a
        high-priority case — never cancel-without-refund silently.

        Regression for the bug where _handle_cancel_item swallowed
        RefundThresholdError into a note and created no case, unlike the
        _handle_process_refund path.
        """
        # ORD-78321 line 1 is a Rs.55,000 laptop on a 'processing' order.
        body = _run(client, "Cancel item 1 from ORD-78321.", [
            make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 1}),
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "success"

        # A high-priority CRM case must exist for the escalated refund.
        crm   = json.loads((data_dir / "crm_cases.json").read_text())
        cases = crm.get("cases", [])
        assert any(
            c.get("order_id") == "ORD-78321" and c.get("priority") == "high"
            for c in cases
        ), "high-value cancel did not create a specialist refund case"

        # The customer is told a specialist will follow up — not that it's done.
        resp = body["response"].lower()
        assert any(w in resp for w in ["specialist", "case", "review", "24"])


# ===========================================================================
# Missing data
# ===========================================================================

class TestMissingData:

    def test_missing_office_address_recorded_as_error(self, client, data_dir):
        crm = json.loads((data_dir / "crm_cases.json").read_text())
        for c in crm["customers"]:
            if c["customer_id"] == "CUST-001":
                c["addresses"] = [a for a in c.get("addresses", [])
                                   if a.get("label") != "office"]
        (data_dir / "crm_cases.json").write_text(json.dumps(crm, indent=2))

        body = _run(client, "Ship to my office address for ORD-78321.", [
            make_tool_mock("update_address", {"order_id": "ORD-78321", "address_label": "office"}),
            make_text_mock("Sorry, I couldn't find your office address."),
        ])
        ac = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "update_address"]
        assert ac and ac[0]["status"] == "error"

    def test_kb_search_empty_tags_returns_empty(self, patched_env):
        from tools.kb_tool import KbTool
        result = asyncio.get_event_loop().run_until_complete(KbTool().search(tags=[]))
        assert result == []

    def test_kb_search_no_matching_tags_returns_empty(self, patched_env):
        from tools.kb_tool import KbTool
        result = asyncio.get_event_loop().run_until_complete(
            KbTool().search(tags=["zzz_no_such_tag_xyz"]))
        assert result == []


# ===========================================================================
# Payment gateway retry
# ===========================================================================

class TestPaymentGatewayRetry:

    def test_gateway_timeout_then_success(self, patched_env):
        # The single retry loop (inside _call_gateway_with_retry) recovers from a
        # transient timeout: first gateway attempt fails (random < failure_rate),
        # second attempt succeeds.
        from tools.payment_tool import PaymentTool

        tool = PaymentTool()
        tool._config["behaviour"]["failure_rate"] = 0.5

        with patch("tools.payment_tool.random.random", side_effect=[0.0, 1.0]):
            result = asyncio.get_event_loop().run_until_complete(
                tool.process_refund("ORD-78321", 1500.0, "HDFC_CREDIT", "CUST-001"))

        assert result["status"]   == "initiated"
        assert result["order_id"] == "ORD-78321"

    def test_all_retries_exhausted_raises_error(self, patched_env):
        from tools.payment_tool import PaymentTool, PaymentGatewayError

        tool = PaymentTool()

        async def always_fail(*a, **k):
            raise PaymentGatewayError("Gateway failed after all retries.")

        with patch.object(tool, "_call_gateway_with_retry", side_effect=always_fail):
            with pytest.raises(PaymentGatewayError):
                asyncio.get_event_loop().run_until_complete(
                    tool.process_refund("ORD-78321", 1500.0, "HDFC_CREDIT", "CUST-001"))

    def test_retry_loop_exhausts_attempts(self, patched_env):
        import sys, tools.payment_tool as pt_mod
        from tools.payment_tool import PaymentTool, PaymentGatewayError

        tool = PaymentTool()
        tool._config["behaviour"]["failure_rate"] = 1.0

        original       = pt_mod.MAX_RETRIES
        original_delay = pt_mod.RETRY_BASE_DELAY_S
        try:
            pt_mod.MAX_RETRIES        = 2
            pt_mod.RETRY_BASE_DELAY_S = 0.0
            with patch("tools.payment_tool.random.random", return_value=0.0):
                with pytest.raises(PaymentGatewayError):
                    asyncio.get_event_loop().run_until_complete(
                        tool._call_gateway_with_retry(
                            "ORD-78321", 1500.0, "HDFC_CREDIT", "CUST-001"))
        finally:
            pt_mod.MAX_RETRIES        = original
            pt_mod.RETRY_BASE_DELAY_S = original_delay


# ===========================================================================
# Refund-method backstop (unsupported method must not silently become 'original')
# ===========================================================================

class TestRefundMethodBackstop:

    def test_resolver_defaults_and_aliases(self, patched_env):
        from agent.graph import _resolve_requested_refund_method as r
        assert r(None)         == ("original", None)
        assert r("original")   == ("original", None)
        assert r("same card")  == ("original", None)   # alias → original
        assert r("hdfc")       == ("HDFC_CREDIT", None)
        assert r("UPI")        == ("UPI", None)

    def test_resolver_rejects_unsupported_method(self, patched_env):
        from agent.graph import _resolve_requested_refund_method as r
        for bad in ("american express", "amex", "AMEX_CREDIT", "crypto", "paypal"):
            method, error = r(bad)
            assert method is None, f"{bad!r} should not resolve to a method"
            assert error and "not supported" in error
            # Non-COD default: original payment method is offered first.
            assert "original payment method" in error
            assert error.index("original payment method") < error.index(":"), \
                "original method must be offered before the electronic list"

    def test_resolver_cod_lists_electronic_only(self, patched_env):
        """COD has no electronic original — the menu must NOT offer 'original',
        only the supported electronic rails."""
        from agent.graph import _resolve_requested_refund_method as r
        method, error = r("american express", original_method="COD")
        assert method is None
        assert "original payment method" not in error
        assert "electronic" in error

    def test_detect_unsupported_method_in_message(self, patched_env):
        from agent.graph import _detect_unsupported_refund_method as d
        # Fires on plainly-unsupported brands phrased as a refund destination.
        assert d("cancel the remaining item and refund me to American Express Card")
        assert d("refund me to my amex")
        assert d("please send the refund to PayPal")
        assert d("refund to my Visa card please")
        # Does NOT fire: supported rail / original named, or no destination phrase.
        assert d("refund to my HDFC credit card") is None
        assert d("refund to my original payment method") is None
        assert d("cancel item 1 and refund to UPI") is None
        assert d("I want a refund of Rs.24999 for order ORD-78323") is None
        assert d("not amex, use my hdfc card") is None

    def test_unsupported_method_message_blocked_without_confirmation(self, client):
        """A customer asking to be refunded to an unsupported method gets the menu
        DIRECTLY — no tool call, no confirmation prompt — driven by their own words."""
        body = _run(client, "Cancel item 1 from ORD-78321 and refund me to American Express Card.", [
            # No planner/responder mock should be consumed — the guard short-circuits.
            make_text_mock("unused"),
        ])
        # No mutating tool ran and no confirmation was requested.
        assert "cancel_item" not in _actions(body)
        assert "request_confirmation" not in _actions(body)
        resp = body["response"].lower()
        assert "american express" in resp
        assert "original payment method" in resp  # offered first (non-COD order)

    def test_refund_failure_does_not_cancel_item(self, client, data_dir):
        """ATOMICITY: if the refund can't be initiated, the item must NOT be
        cancelled — no cancelled-without-refund orphan (regression for ORD-10001
        line 3, which was cancelled with no refund record/case/audit)."""
        import agent.graph as g
        from tools.payment_tool import PaymentGatewayError

        async def boom(*a, **k):
            raise PaymentGatewayError("gateway unavailable")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 2}),
            make_text_mock("Sorry, I couldn't process that right now."),
            make_approved_mock(),
        ])
        with patch("agent.graph._groq_client", mock_client):
            with patch.object(g._payment, "process_refund", side_effect=boom):
                resp = client.post("/query", json={
                    "message": "Cancel item 2 from ORD-78321.", "session_id": "sess-cust001",
                })
                body = resp.json()

        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error", "refund failure must surface as an error"

        # The item must still be active — never cancelled without a refund trace.
        orders = json.loads((data_dir / "orders.json").read_text())
        order  = next(o for o in orders["orders"] if o["order_id"] == "ORD-78321")
        item2  = next(i for i in order["items"] if i["line_id"] == 2)
        assert item2["status"] != "cancelled", "item was cancelled despite refund failure"

    def test_process_refund_unsupported_method_blocked(self, client):
        """Handler-layer backstop: if an unsupported method reaches process_refund
        (e.g. a name the message guard doesn't recognise), it errors and disburses
        no refund — never silently coerced to an 'original' refund. Neutral message
        so the deterministic message-level guard does not pre-empt the handler."""
        body = _run(client, "Process the refund for order ORD-78323.", [
            make_tool_mock("process_refund", {
                "order_id": "ORD-78323", "amount_inr": 1000.0, "method": "gift card",
            }),
            make_text_mock("I can refund to your original method or a supported one."),
            make_approved_mock(),
        ])
        pr = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "process_refund"]
        assert pr and pr[0]["status"] == "error"

    def test_cancel_item_unsupported_method_blocked(self, client):
        """Handler-layer backstop: cancel_item with an unsupported refund_method
        errors BEFORE cancelling, so the item is never cancelled-without-a-valid-
        refund. Neutral message so the message-level guard does not pre-empt it."""
        body = _run(client, "Cancel item 2 from ORD-78321.", [
            make_tool_mock("cancel_item", {
                "order_id": "ORD-78321", "line_id": 2, "refund_method": "gift card",
            }),
            make_text_mock("We can't refund to that — please choose a supported method."),
            make_approved_mock(),
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_cancel_item_missing_line_id_clean_error(self, client):
        """A cancel_item call with NO line_id (e.g. a confirmation staged without
        it) must return a clean error, never crash with KeyError → 500."""
        body = _run(client, "Cancel an item on ORD-78321.", [
            make_tool_mock("cancel_item", {"order_id": "ORD-78321"}),  # line_id omitted
            make_text_mock("Which item would you like to cancel?"),
            make_approved_mock(),
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_cancel_item_invalid_line_id_clean_error(self, client):
        """A non-integer line_id must surface a clean error, not an exception."""
        body = _run(client, "Cancel an item on ORD-78321.", [
            make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": "abc"}),
            make_text_mock("That doesn't look like a valid item."),
            make_approved_mock(),
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"


class TestOrderIdNormalization:
    """A bare 5-digit order number must resolve to canonical ORD-XXXXX for tools."""

    def test_clean_order_id_normalises_bare_number(self, patched_env):
        from agent.graph import _clean_order_id as c
        assert c("10001")                       == "ORD-10001"
        assert c("order 10001")                 == "ORD-10001"
        assert c("#10001")                      == "ORD-10001"
        assert c("pull up info about 10001")    == "ORD-10001"
        # Already-canonical and edge cases must be untouched.
        assert c("ORD-10001")                   == "ORD-10001"
        assert c("ORD-123456")                  == "ORD-123456"  # 6 digits: never truncated
        assert c("1234")                        == "1234"        # 4 digits: not an order number

    def test_get_order_resolves_bare_number(self, client):
        """'pull up info about 78321' → get_order('78321') resolves ORD-78321."""
        body = _run(client, "pull up info about 78321", [
            make_tool_mock("get_order", {"order_id": "78321"}),
            make_text_mock("Here are the details for your order."),
            make_approved_mock(),
        ])
        go = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "get_order"]
        assert go and go[0]["status"] == "success", "bare order number did not resolve"


class TestMixedEscalationResponder:
    """A turn that both completes actions AND escalates must narrate both — the
    escalation holding message must not swallow completed refunds/cancels (Q1 bug)."""

    def test_is_pure_escalation_turn_unit(self, patched_env):
        from agent.graph import _is_pure_escalation_turn as p
        mixed = [
            {"tool": "process_refund", "success": True, "escalated": False,
             "data": {"refund": {"amount_inr": 800}}},
            {"tool": "process_refund", "success": True, "escalated": True,
             "data": {"case_id": "CASE-X"}},
        ]
        pure = [
            {"tool": "get_order", "success": True, "escalated": False, "data": {}},
            {"tool": "process_refund", "success": True, "escalated": True,
             "data": {"case_id": "CASE-Y"}},
        ]
        assert p(mixed) is False, "mixed turn must use the full responder"
        assert p(pure) is True, "pure escalation uses the deterministic holding message"
        assert p([]) is False

    def test_mixed_turn_uses_full_responder(self, client):
        """Disburse a refund on one delivered order AND escalate another (over-limit)
        in one turn → the LLM responder is used, not the canned escalation text."""
        custom = "Your Rs.800 refund is on its way; the high-value item is under specialist review."
        body = _run(client, "Refund the mouse on ORD-78323 and the gaming laptop on ORD-78500.", [
            make_multi_tool_mock([
                ("process_refund", {"order_id": "ORD-78323", "amount_inr": 800.0, "method": "original"}),
                ("process_refund", {"order_id": "ORD-78500", "amount_inr": 42000.0, "method": "original"}),
            ]),
            make_text_mock(custom),
            make_approved_mock(),
        ])
        pr = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "process_refund"]
        assert len(pr) == 2, "both refunds should have been attempted"
        # The mixed turn must NOT collapse to the deterministic escalation holding text.
        assert body["response"] == custom
        assert "apologise for any inconvenience" not in body["response"].lower()

    def test_pure_escalation_uses_canned_message(self, client):
        """A turn that ONLY escalates still gets the deterministic holding message
        (no LLM responder call)."""
        body = _run(client, "Refund my gaming laptop on ORD-78500.", [
            make_tool_mock("process_refund", {"order_id": "ORD-78500", "amount_inr": 42000.0, "method": "original"}),
            make_approved_mock(),  # spare; pure-escalation path is deterministic
        ])
        resp = body["response"].lower()
        assert "specialist review" in resp or "apologise for any inconvenience" in resp


# ===========================================================================
# LLM resilience
# ===========================================================================

class TestLLMResilience:

    def test_llm_api_failure_returns_safe_response(self, client):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API error"))
        with patch("agent.graph._groq_client", mock_client):
            resp = client.post("/query",
                json={"message": "Where is my order?", "session_id": "sess-cust001"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"]
        assert "traceback"  not in body["response"].lower()
        assert "exception"  not in body["response"].lower()

    def test_llm_timeout_returns_safe_response(self, client):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=asyncio.TimeoutError("LLM timed out"))
        with patch("agent.graph._groq_client", mock_client):
            resp = client.post("/query",
                json={"message": "Where is my order?", "session_id": "sess-cust001"})
        assert resp.status_code == 200
        assert resp.json()["response"]

    def test_unknown_message_returns_safe_response(self, client):
        body = _run(client, "Do something completely weird.", [
            make_done_mock("I can't help with that."),
        ])
        assert body["response"]


# ===========================================================================
# Hallucination prevention
# ===========================================================================

class TestHallucinationPrevention:

    def test_response_does_not_invent_tracking_number(self, client):
        body = _run(client, "Track ORD-78324.", [
            make_tool_mock("get_order", {"order_id": "ORD-78324"}),
            make_text_mock("Your order ORD-78324 has been placed and is being prepared."),
        ])
        fabricated = re.findall(r"TRACK-[A-Z0-9]+", body["response"])
        assert not fabricated

    def test_nonexistent_order_response_has_no_invented_data(self, client):
        body = _run(client, "Where is ORD-00000?", [
            make_tool_mock("get_order", {"order_id": "ORD-00000"}),
            make_text_mock("I'm sorry, I couldn't find order ORD-00000."),
        ])
        resp = body["response"].lower()
        assert not ("delivered" in resp and "not" not in resp)


# ===========================================================================
# Trace integrity
# ===========================================================================

class TestTraceIntegrity:

    def test_trace_id_unique_per_request(self, client):
        b1 = _run(client, "Track ORD-78321.", [
            make_tool_mock("get_order", {"order_id": "ORD-78321"}),
            make_text_mock("Processing."),
        ])
        b2 = _run(client, "Track ORD-78321.", [
            make_tool_mock("get_order", {"order_id": "ORD-78321"}),
            make_text_mock("Processing."),
        ])
        assert b1["trace"]["trace_id"] != b2["trace"]["trace_id"]

    def test_latency_ms_always_present_and_positive(self, client):
        body = _run(client, "Track ORD-78321.", [
            make_tool_mock("get_order", {"order_id": "ORD-78321"}),
            make_text_mock("Processing."),
        ])
        assert isinstance(body["trace"]["latency_ms"], int)
        assert body["trace"]["latency_ms"] > 0

    def test_failed_step_appears_in_trace(self, client):
        body = _run(client, "Track ORD-00000.", [
            make_tool_mock("get_order", {"order_id": "ORD-00000"}),
            make_text_mock("Not found."),
        ])
        calls = body["trace"]["tool_calls"]
        assert len(calls) >= 1
        assert any(s != "success" for s in _statuses(body))

    def test_trace_tool_calls_ordered_correctly(self, client):
        body  = _run(client, "Track ORD-78321.", [
            make_tool_mock("get_order", {"order_id": "ORD-78321"}),
            make_text_mock("Processing."),
        ])
        tools = [tc["tool"] for tc in body["trace"]["tool_calls"]]
        if "agent_70b" in tools and "get_order" in tools:
            assert tools.index("agent_70b") < tools.index("get_order")


# ===========================================================================
# Validators
# ===========================================================================

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
