"""
tests/test_journeys.py
=======================
End-to-end journey tests for J1, J2, and J3.
"""

import json
import time
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from tests.conftest import J1_PLAN, J2_PLAN, J3_PLAN, _mock_plan_response, make_llm_mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(client, message, session_id="sess-cust001"):
    resp = client.post("/query", json={"message": message, "session_id": session_id})
    assert resp.status_code == 200, f"Unexpected {resp.status_code}: {resp.text}"
    return resp.json()


def _actions(body):  return [tc["action"] for tc in body["trace"]["tool_calls"]]
def _statuses(body): return [tc["status"] for tc in body["trace"]["tool_calls"]]
def _tools(body):    return [tc["tool"]   for tc in body["trace"]["tool_calls"]]


def _run(client, message, plan, resp_text="Done.", session_id="sess-cust001"):
    """Post a query with both LLM calls mocked. Returns response body."""
    mock = make_llm_mock(plan, resp_text)
    with patch("agent.planner.Planner._call_llm", new=mock):
        with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
            return _post(client, message, session_id)


# ===========================================================================
# J1 — Order Tracking
# ===========================================================================

class TestJ1OrderTracking:

    def test_j1_returns_200(self, client):
        body = _run(client, "Where is my order ORD-78321?", J1_PLAN)
        assert body["response"]
        assert body["trace"]["trace_id"].startswith("trc-")

    def test_j1_latency_under_3_seconds(self, client):
        mock = make_llm_mock(J1_PLAN, "Your order is being processed.")
        with patch("agent.planner.Planner._call_llm", new=mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
                t0 = time.monotonic()
                _post(client, "Where is my order ORD-78321?")
                elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 3000, f"J1 latency {elapsed_ms:.0f}ms exceeds 3000ms SLA"

    def test_j1_exactly_one_oms_call(self, client):
        body = _run(client, "Where is my order ORD-78321?", J1_PLAN)
        oms_calls = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "get_order"]
        assert len(oms_calls) == 1, f"Expected exactly 1 OMS call, got {len(oms_calls)}: {oms_calls}"

    def test_j1_no_crm_or_payment_calls(self, client):
        body = _run(client, "Where is my order ORD-78321?", J1_PLAN)
        acts = _actions(body)
        assert "process_refund"  not in acts
        assert "create_crm_case" not in acts
        assert "escalate"        not in acts

    def test_j1_no_escalation(self, client):
        body = _run(client, "Where is my order ORD-78321?", J1_PLAN)
        assert "escalate" not in _actions(body)

    def test_j1_trace_fields_present(self, client):
        body = _run(client, "Where is my order ORD-78321?", J1_PLAN)
        trace = body["trace"]
        assert "trace_id"   in trace
        assert "session_id" in trace
        assert "latency_ms" in trace
        assert "tool_calls" in trace
        assert trace["session_id"] == "sess-cust001"
        assert isinstance(trace["latency_ms"], int)
        assert trace["latency_ms"] > 0

    def test_j1_shipped_order_returns_tracking(self, client):
        plan = _mock_plan_response("order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78322"}, "depends_on": []}])
        body = _run(client, "Where is my order ORD-78322?", plan, "Shipped with TRACK-7X9K2M.")
        assert "success" in _statuses(body)

    def test_j1_response_is_non_empty(self, client):
        body = _run(client, "Where is my order ORD-78321?", J1_PLAN)
        assert len(body["response"]) > 10

    def test_j1_response_not_truncated(self, client):
        body = _run(client, "Where is my order ORD-78321?", J1_PLAN)
        assert body["response"].strip()[-1] in ".?!*)"


# ===========================================================================
# J2 — Multi-Step Compound
# ===========================================================================

class TestJ2CompoundRequest:

    def test_j2_all_three_steps_in_trace(self, client):
        body = _run(
            client,
            "Cancel item 2 from ORD-78321, refund to HDFC card, ship rest to office.",
            J2_PLAN,
        )
        acts = _actions(body)
        assert "cancel_item"    in acts, f"cancel_item not in {acts}"
        assert "process_refund" in acts, f"process_refund not in {acts}"
        assert "update_address" in acts, f"update_address not in {acts}"

    def test_j2_cancel_step_recorded_as_success(self, client):
        body = _run(client, "Cancel item 2 from ORD-78321.", J2_PLAN)
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert len(cc) >= 1
        assert cc[0]["status"] == "success"

    def test_j2_refund_step_recorded_as_success(self, client):
        body = _run(client, "Cancel item 2 from ORD-78321 and refund.", J2_PLAN)
        rc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "process_refund"]
        assert len(rc) >= 1
        assert rc[0]["status"] == "success"

    def test_j2_address_update_step_recorded(self, client):
        body = _run(client, "Ship remaining items to office for ORD-78321.", J2_PLAN)
        ac = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "update_address"]
        assert len(ac) >= 1

    def test_j2_failed_step_recorded_not_hidden(self, client):
        plan = _mock_plan_response("compound", [
            {"action": "cancel_item",
             "params": {"order_id": "ORD-78323", "line_id": 1},
             "depends_on": []},
        ])
        body = _run(client, "Cancel item 1 from ORD-78323.", plan)
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert len(cc) >= 1
        assert cc[0]["status"] == "error"

    def test_j2_dependent_step_skipped_on_cancel_failure(self, client):
        plan = _mock_plan_response("compound", [
            {"action": "cancel_item",
             "params": {"order_id": "ORD-78323", "line_id": 1},
             "depends_on": []},
            {"action": "process_refund",
             "params": {"order_id": "ORD-78323", "amount_inr": 55000.0, "method": "HDFC_CREDIT"},
             "depends_on": [0]},
        ])
        body = _run(client, "Cancel and refund ORD-78323.", plan)
        rc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "process_refund"]
        if rc:
            assert rc[0]["status"] != "success"

    def test_j2_returns_200(self, client):
        body = _run(client, "Cancel item 2 from ORD-78321.", J2_PLAN)
        assert body["response"]


# ===========================================================================
# J3 — Escalation with Audit Trail
# ===========================================================================

class TestJ3Escalation:

    def _run_j3(self, client, message="Laptop arrived damaged, order ORD-78500."):
        """
        Run a J3 escalation request with guardrail GR-001 bypassed.

        GR-001 fires pre-LLM when it detects a refund amount > Rs.25,000
        in the message. For J3 tests, the plan itself contains the amount
        (42000) but the natural language message does NOT mention the amount
        — so GR-001 does not fire and the escalate step runs normally.

        Alternatively we patch guardrails.pre_check to allow for tests
        that do include the amount.
        """
        from agent.guardrails import GuardrailVerdict
        mock = make_llm_mock(J3_PLAN, "Your case has been escalated.")
        with patch("agent.planner.Planner._call_llm", new=mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=mock):
                with patch(
                    "agent.guardrails.Guardrails.pre_check",
                    return_value=GuardrailVerdict.allow(),
                ):
                    return _post(client, message)

    def test_j3_payment_tool_never_called(self, client):
        body = self._run_j3(client)
        assert "process_refund" not in _actions(body), (
            f"CRITICAL: process_refund called on escalation! "
            f"tool_calls: {body['trace']['tool_calls']}"
        )

    def test_j3_crm_case_created(self, client, data_dir):
        self._run_j3(client)
        crm = json.loads((data_dir / "crm_cases.json").read_text())
        assert len(crm["cases"]) >= 1, "No CRM case was created."

    def test_j3_case_has_trace_id(self, client, data_dir):
        body = self._run_j3(client)
        tid  = body["trace"]["trace_id"]
        crm  = json.loads((data_dir / "crm_cases.json").read_text())
        assert tid in [c.get("trace_id") for c in crm["cases"]], (
            f"trace_id '{tid}' not found in cases."
        )

    def test_j3_case_has_structured_handoff(self, client, data_dir):
        self._run_j3(client)
        crm  = json.loads((data_dir / "crm_cases.json").read_text())
        assert crm["cases"], "No case created."
        desc = crm["cases"][0]["description"]
        assert "[ESCALATION CASE" in desc
        assert "CUST-001"         in desc
        assert "ORD-78500"        in desc
        assert "Trace ID"         in desc

    def test_j3_case_priority_is_high(self, client, data_dir):
        self._run_j3(client)
        crm = json.loads((data_dir / "crm_cases.json").read_text())
        assert crm["cases"][0]["priority"] == "high"

    def test_j3_user_gets_polite_holding_message(self, client):
        body = self._run_j3(client)
        r    = body["response"].lower()
        assert "error"     not in r
        assert "exception" not in r
        assert any(k in r for k in ["case", "specialist", "team", "24", "escalat"])

    def test_j3_guardrail_pre_check_blocks_high_value(self, client):
        # No LLM mock needed — GR-001 fires before planner on amount mention
        body = _post(client, "Please refund Rs.42000 for my damaged laptop.")
        guardrail_calls = [tc for tc in body["trace"]["tool_calls"]
                           if tc["tool"] == "guardrails"]
        assert len(guardrail_calls) >= 1
        assert "process_refund" not in _actions(body)

    def test_j3_case_id_matches_schema_pattern(self, client, data_dir):
        import re
        self._run_j3(client)
        crm     = json.loads((data_dir / "crm_cases.json").read_text())
        case_id = crm["cases"][0]["case_id"]
        assert re.match(r"^CASE-[A-Z0-9]{6}$", case_id), (
            f"case_id '{case_id}' does not match schema pattern."
        )

    def test_j3_escalation_in_trace(self, client):
        body = self._run_j3(client)
        # The escalate action is recorded with tool=action.value="escalate"
        assert "escalate" in _actions(body), (
            f"Expected 'escalate' in actions, got: {_actions(body)}"
        )

    def test_j3_returns_200(self, client):
        body = self._run_j3(client)
        assert body["response"]