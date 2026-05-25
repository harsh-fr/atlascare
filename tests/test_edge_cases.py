"""
tests/test_edge_cases.py
=========================
Edge cases, failure modes, and hallucination prevention tests.
"""

import asyncio
import json
import re
import pytest
from unittest.mock import patch, AsyncMock
from tests.conftest import _mock_plan_response, make_llm_mock, _make_order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(client, message, plan, resp="Done.", session_id="sess-cust001"):
    mock = make_llm_mock(plan, resp)
    with patch("agent.planner.Planner._call_llm", new=mock):
        with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
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
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}])
        body = _run(client, "Where is order ORD-00000?", plan)
        s    = _statuses(body)
        assert "error" in s or "ownership_denied" in s

    def test_nonexistent_order_returns_200_not_500(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}])
        body = _run(client, "Track ORD-00000.", plan)
        assert body  # 200 asserted inside _run

    def test_cancel_nonexistent_line_id_recorded_as_error(self, client):
        plan = _mock_plan_response("partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-78321", "line_id": 99},
              "depends_on": []}])
        body = _run(client, "Cancel item 99 from ORD-78321.", plan)
        cc   = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_cancel_already_cancelled_item_recorded_as_error(self, client, data_dir):
        """
        Cancelling an already-cancelled item must record status=error.

        Strategy: mock OrderRepository.find_by_id to always return an
        order where item 1 is already cancelled. This bypasses the
        in-memory index already-loaded by the app at startup.
        """
        import copy
        from repositories.order_repository import OrderRepository

        # Build a pre-cancelled version of ORD-78321
        base_order = _make_order("ORD-78321", "CUST-001", "processing")
        cancelled_order = copy.deepcopy(base_order)
        for i in cancelled_order["items"]:
            if i["line_id"] == 1:
                i["status"] = "cancelled"

        original_find = OrderRepository.find_by_id

        def patched_find(self_repo, order_id):
            if order_id.upper() == "ORD-78321":
                return copy.deepcopy(cancelled_order)
            return original_find(self_repo, order_id)

        plan = _mock_plan_response("partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-78321", "line_id": 1},
              "depends_on": []}])
        mock = make_llm_mock(plan, "Already cancelled.")

        with patch("agent.planner.Planner._call_llm", new=mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
                with patch.object(OrderRepository, "find_by_id", patched_find):
                    resp = client.post("/query",
                        json={"message": "Cancel item 1 from ORD-78321.",
                              "session_id": "sess-cust001"})

        assert resp.status_code == 200
        body = resp.json()
        cc   = [tc for tc in body["trace"]["tool_calls"]
                if tc["action"] == "cancel_item"]
        assert cc, "cancel_item not found in tool_calls"
        assert cc[0]["status"] == "error", (
            f"Expected error for already-cancelled item, got: {cc[0]['status']}"
        )

    def test_cancel_shipped_order_recorded_as_error(self, client):
        plan = _mock_plan_response("partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-78322", "line_id": 1},
              "depends_on": []}])
        body = _run(client, "Cancel item 1 from ORD-78322.", plan)
        cc   = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_cancel_delivered_order_recorded_as_error(self, client):
        plan = _mock_plan_response("partial_cancellation",
            [{"action": "cancel_item",
              "params": {"order_id": "ORD-78323", "line_id": 1},
              "depends_on": []}])
        body = _run(client, "Cancel item 1 from ORD-78323.", plan)
        cc   = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"


# ===========================================================================
# Missing data
# ===========================================================================

class TestMissingData:

    def test_missing_office_address_recorded_as_error(self, client, data_dir):
        # Remove office address from CUST-001
        crm = json.loads((data_dir / "crm_cases.json").read_text())
        for c in crm["customers"]:
            if c["customer_id"] == "CUST-001":
                c["addresses"] = [a for a in c.get("addresses", [])
                                   if a.get("label") != "office"]
        (data_dir / "crm_cases.json").write_text(json.dumps(crm, indent=2))

        plan = _mock_plan_response("address_update",
            [{"action": "update_address",
              "params": {"order_id": "ORD-78321", "address_label": "office"},
              "depends_on": []}])
        body = _run(client, "Ship to my office address for ORD-78321.", plan)
        ac   = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "update_address"]
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
        """
        Gateway fails once then succeeds — final result must be success.

        Approach: mock _call_gateway_with_retry at the instance level so
        we are completely independent of MAX_RETRIES, RETRY_BASE_DELAY_S,
        and failure_rate. First call raises, second call returns success.
        This is the correct way to test retry behaviour without fighting
        module-level constant resolution.
        """
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

        # process_refund calls _call_gateway_with_retry internally.
        # We patch it on the instance so only this tool is affected.
        with patch.object(tool, "_call_gateway_with_retry", side_effect=retry_mock):
            result = asyncio.get_event_loop().run_until_complete(
                tool.process_refund("ORD-78321", 1500.0, "HDFC_CREDIT", "CUST-001"))

        assert result["status"]    == "initiated"
        assert result["refund_id"] == "REF-78321-TEST0001"

    def test_all_retries_exhausted_raises_error(self, patched_env):
        """
        All retries exhausted → PaymentGatewayError propagates out of process_refund.

        Approach: mock _call_gateway_with_retry to always raise so we test
        that PaymentTool.process_refund correctly propagates the error.
        """
        from tools.payment_tool import PaymentTool, PaymentGatewayError

        tool = PaymentTool()

        async def always_fail(*a, **k):
            raise PaymentGatewayError("Gateway failed after all retries.")

        with patch.object(tool, "_call_gateway_with_retry", side_effect=always_fail):
            with pytest.raises(PaymentGatewayError):
                asyncio.get_event_loop().run_until_complete(
                    tool.process_refund(
                        "ORD-78321", 1500.0, "HDFC_CREDIT", "CUST-001"
                    )
                )

    def test_retry_loop_exhausts_attempts(self, patched_env):
        """
        Unit test of _call_gateway_with_retry loop directly.
        Sets failure_rate=1.0 on the config and MAX_RETRIES=2 via sys.modules
        so every attempt hits the timeout branch.
        """
        import sys, tools.payment_tool as pt_mod
        from tools.payment_tool import PaymentTool, PaymentGatewayError

        tool = PaymentTool()
        tool._config["behaviour"]["failure_rate"] = 1.0

        # Directly mutate the module attribute — patch.object does the same
        original = pt_mod.MAX_RETRIES
        original_delay = pt_mod.RETRY_BASE_DELAY_S
        try:
            pt_mod.MAX_RETRIES = 2
            pt_mod.RETRY_BASE_DELAY_S = 0.0
            with patch("tools.payment_tool.random.random", return_value=0.0):
                with pytest.raises(PaymentGatewayError):
                    asyncio.get_event_loop().run_until_complete(
                        tool._call_gateway_with_retry(
                            "ORD-78321", 1500.0, "HDFC_CREDIT", "CUST-001"
                        )
                    )
        finally:
            pt_mod.MAX_RETRIES = original
            pt_mod.RETRY_BASE_DELAY_S = original_delay


# ===========================================================================
# Planner resilience
# ===========================================================================

class TestPlannerResilience:

    def test_malformed_llm_json_returns_safe_response(self, client):
        with patch("agent.planner.Planner._call_llm",
                   new=AsyncMock(return_value="Not JSON at all!!!")):
            resp = client.post("/query",
                json={"message": "Where is my order?", "session_id": "sess-cust001"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"]
        assert "traceback"  not in body["response"].lower()
        assert "exception"  not in body["response"].lower()

    def test_unknown_intent_returns_safe_response(self, client):
        plan = _mock_plan_response("totally_unknown_intent_xyz", [])
        body = _run(client, "Do something weird.", plan)
        assert body["response"]

    def test_llm_timeout_returns_safe_response(self, client):
        async def timeout_llm(*a, **k):
            raise asyncio.TimeoutError("LLM timed out")
        with patch("agent.planner.Planner._call_llm", new=timeout_llm):
            resp = client.post("/query",
                json={"message": "Where is my order?", "session_id": "sess-cust001"})
        assert resp.status_code == 200
        assert resp.json()["response"]


# ===========================================================================
# Hallucination prevention
# ===========================================================================

class TestHallucinationPrevention:

    def test_response_does_not_invent_tracking_number(self, client):
        # ORD-78324 is 'placed' — tracking_number is None
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78324"}, "depends_on": []}])
        body = _run(client, "Track ORD-78324.", plan,
                    "Your order ORD-78324 has been placed and is being prepared.")
        fabricated = re.findall(r"TRACK-[A-Z0-9]+", body["response"])
        assert not fabricated, f"Response contains fabricated tracking: {fabricated}"

    def test_nonexistent_order_response_has_no_invented_data(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}])
        body = _run(client, "Where is ORD-00000?", plan)
        resp = body["response"].lower()
        # Must not invent a positive status for a non-existent order
        assert not ("delivered" in resp and "not" not in resp)


# ===========================================================================
# Trace integrity
# ===========================================================================

class TestTraceIntegrity:

    def test_trace_id_unique_per_request(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}])
        b1 = _run(client, "Track ORD-78321.", plan)
        b2 = _run(client, "Track ORD-78321.", plan)
        assert b1["trace"]["trace_id"] != b2["trace"]["trace_id"]

    def test_latency_ms_always_present_and_positive(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}])
        body = _run(client, "Track ORD-78321.", plan)
        assert isinstance(body["trace"]["latency_ms"], int)
        assert body["trace"]["latency_ms"] > 0

    def test_failed_step_appears_in_trace(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-00000"}, "depends_on": []}])
        body = _run(client, "Track ORD-00000.", plan)
        calls = body["trace"]["tool_calls"]
        assert len(calls) >= 1
        assert any(s != "success" for s in _statuses(body))

    def test_trace_tool_calls_ordered_correctly(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78321"}, "depends_on": []}])
        body  = _run(client, "Track ORD-78321.", plan)
        tools = [tc["tool"] for tc in body["trace"]["tool_calls"]]
        if "planner" in tools and "get_order" in tools:
            assert tools.index("planner") < tools.index("get_order")


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