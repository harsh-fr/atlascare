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
from tests.conftest import make_tool_mock, make_done_mock, make_text_mock, _make_order


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
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_cancel_delivered_order_recorded_as_error(self, client):
        body = _run(client, "Cancel item 1 from ORD-78323.", [
            make_tool_mock("cancel_item", {"order_id": "ORD-78323", "line_id": 1}),
            make_text_mock("That order is delivered and cannot be cancelled."),
        ])
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"


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
        from tools.payment_tool import PaymentTool, PaymentGatewayError

        tool    = PaymentTool()
        call_n  = 0
        success = {
            "refund_id": "REF-78321-TEST0001", "order_id": "ORD-78321",
            "amount_inr": 1500.0, "method": "HDFC_CREDIT",
            "status": "initiated", "sla_days": 5, "message": "Refund initiated.",
        }

        async def retry_mock(*a, **k):
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                raise PaymentGatewayError("Simulated timeout on attempt 1.")
            return success

        with patch.object(tool, "_call_gateway_with_retry", side_effect=retry_mock):
            result = asyncio.get_event_loop().run_until_complete(
                tool.process_refund("ORD-78321", 1500.0, "HDFC_CREDIT", "CUST-001"))

        assert result["status"]    == "initiated"
        assert result["refund_id"] == "REF-78321-TEST0001"

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
