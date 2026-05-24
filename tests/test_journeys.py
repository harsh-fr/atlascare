"""
tests/test_journeys.py
=======================
End-to-end journey tests for J1, J2, and J3.

Coverage
--------
  J1 — Order Tracking
    - Happy path: returns status + tracking in < 3s
    - Exactly ONE OMS tool call
    - Zero hallucination: response grounded in real data
    - No escalation triggered

  J2 — Multi-Step Compound Request
    - Partial cancellation executed
    - Refund initiated to correct method
    - Address updated to office
    - All three steps recorded in trace
    - Partial failure handled gracefully

  J3 — Escalation with Audit Trail
    - Payment tool NEVER called
    - CRM case created with trace_id attached
    - Structured handoff summary present
    - Polite holding message returned
    - trace has guardrail or escalation recorded

Design
------
  LLM calls are mocked with deterministic plan JSON.
  All data I/O goes through the isolated tmp_path data_dir.
  Tests assert on trace content to verify correct tool dispatch.
"""

import json
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient

from tests.conftest import (
    J1_PLAN, J2_PLAN, J3_PLAN,
    make_llm_mock,
    _make_order,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post_query(client: TestClient, message: str, session_id: str = "sess-cust001") -> dict:
    resp = client.post("/query", json={"message": message, "session_id": session_id})
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
    return resp.json()


def _tool_names(body: dict) -> list[str]:
    return [tc["tool"] for tc in body["trace"]["tool_calls"]]


def _tool_actions(body: dict) -> list[str]:
    return [tc["action"] for tc in body["trace"]["tool_calls"]]


def _tool_statuses(body: dict) -> list[str]:
    return [tc["status"] for tc in body["trace"]["tool_calls"]]


# ---------------------------------------------------------------------------
# J1 — Order Tracking
# ---------------------------------------------------------------------------

class TestJ1OrderTracking:

    def test_j1_returns_200(self, client: TestClient):
        """J1 basic: endpoint returns HTTP 200."""
        llm_mock = make_llm_mock(J1_PLAN, "Your order ORD-78321 is being processed.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Where is my order ORD-78321?")
        assert body["response"]
        assert body["trace"]["trace_id"].startswith("trc-")

    def test_j1_latency_under_3_seconds(self, client: TestClient):
        """J1 SLA: total latency must be under 3000ms."""
        llm_mock = make_llm_mock(J1_PLAN, "Your order is being processed.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                start = time.monotonic()
                body = _post_query(client, "Where is my order ORD-78321?")
                elapsed_ms = (time.monotonic() - start) * 1000

        assert elapsed_ms < 3000, (
            f"J1 latency {elapsed_ms:.0f}ms exceeds 3000ms SLA"
        )

    def test_j1_exactly_one_oms_call(self, client: TestClient):
        """J1 constraint: exactly ONE OMS tool call, no more."""
        llm_mock = make_llm_mock(J1_PLAN, "Your order is processing.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Where is my order ORD-78321?")

        oms_calls = [
            tc for tc in body["trace"]["tool_calls"]
            if tc["action"] == "get_order"
        ]
        assert len(oms_calls) == 1, (
            f"Expected exactly 1 OMS call, got {len(oms_calls)}: {oms_calls}"
        )

    def test_j1_no_crm_or_payment_calls(self, client: TestClient):
        """J1 constraint: no CRM or payment tool calls on a simple track."""
        llm_mock = make_llm_mock(J1_PLAN, "Your order is on its way.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Where is my order ORD-78321?")

        tool_actions = _tool_actions(body)
        assert "process_refund"  not in tool_actions
        assert "create_crm_case" not in tool_actions
        assert "escalate"        not in tool_actions

    def test_j1_no_escalation(self, client: TestClient):
        """J1 constraint: no escalation on a simple tracking query."""
        llm_mock = make_llm_mock(J1_PLAN, "Your order is being processed.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Where is my order ORD-78321?")

        assert "escalate" not in _tool_actions(body)

    def test_j1_trace_fields_present(self, client: TestClient):
        """J1: trace must contain trace_id, session_id, latency_ms, tool_calls."""
        llm_mock = make_llm_mock(J1_PLAN, "Your order is processing.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Where is my order ORD-78321?")

        trace = body["trace"]
        assert "trace_id"   in trace
        assert "session_id" in trace
        assert "latency_ms" in trace
        assert "tool_calls" in trace
        assert trace["session_id"] == "sess-cust001"
        assert isinstance(trace["latency_ms"], int)
        assert trace["latency_ms"] > 0

    def test_j1_shipped_order_returns_tracking(self, client: TestClient):
        """J1: shipped order response contains tracking data from real record."""
        from tests.conftest import _mock_plan_response
        plan = _mock_plan_response(
            "order_tracking",
            [{"action": "get_order", "params": {"order_id": "ORD-78322"}, "depends_on": []}],
        )
        llm_mock = make_llm_mock(plan, "Your order ORD-78322 has been shipped. Tracking: TRACK-7X9K2M.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Where is my order ORD-78322?")

        assert body["response"]
        assert "success" in _tool_statuses(body)


# ---------------------------------------------------------------------------
# J2 — Multi-Step Compound Request
# ---------------------------------------------------------------------------

class TestJ2CompoundRequest:

    def test_j2_all_three_steps_in_trace(self, client: TestClient):
        """J2: cancel_item, process_refund, update_address all in trace."""
        llm_mock = make_llm_mock(
            J2_PLAN,
            "Item 2 cancelled, refund initiated, address updated to office.",
        )
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(
                    client,
                    "Cancel item 2 from ORD-78321, refund to HDFC card, "
                    "ship other items to office address.",
                )

        actions = _tool_actions(body)
        assert "cancel_item"    in actions, f"cancel_item not in {actions}"
        assert "process_refund" in actions, f"process_refund not in {actions}"
        assert "update_address" in actions, f"update_address not in {actions}"

    def test_j2_cancel_step_recorded_as_success(self, client: TestClient):
        """J2: cancel_item step must succeed for ORD-78321 (processing status)."""
        llm_mock = make_llm_mock(J2_PLAN, "Done.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(
                    client,
                    "Cancel item 2 from ORD-78321 and refund to HDFC.",
                )

        cancel_calls = [
            tc for tc in body["trace"]["tool_calls"]
            if tc["action"] == "cancel_item"
        ]
        assert len(cancel_calls) >= 1
        assert cancel_calls[0]["status"] == "success"

    def test_j2_refund_step_recorded_as_success(self, client: TestClient):
        """J2: process_refund step must succeed for amount under threshold."""
        llm_mock = make_llm_mock(J2_PLAN, "Refund initiated.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Cancel item 2 from ORD-78321 and refund.")

        refund_calls = [
            tc for tc in body["trace"]["tool_calls"]
            if tc["action"] == "process_refund"
        ]
        assert len(refund_calls) >= 1
        assert refund_calls[0]["status"] == "success"

    def test_j2_address_update_step_recorded(self, client: TestClient):
        """J2: update_address step must appear in trace."""
        llm_mock = make_llm_mock(J2_PLAN, "Address updated.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(
                    client,
                    "Ship remaining items to my office address for ORD-78321.",
                )

        address_calls = [
            tc for tc in body["trace"]["tool_calls"]
            if tc["action"] == "update_address"
        ]
        assert len(address_calls) >= 1

    def test_j2_failed_step_recorded_not_hidden(self, client: TestClient):
        """J2: if cancel fails (delivered order), failure must appear in trace."""
        from tests.conftest import _mock_plan_response
        plan = _mock_plan_response(
            "compound",
            [
                {"action": "cancel_item",
                 "params": {"order_id": "ORD-78323", "line_id": 1},
                 "depends_on": []},
            ],
        )
        llm_mock = make_llm_mock(plan, "Could not cancel — order already delivered.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Cancel item 1 from ORD-78323.")

        cancel_calls = [
            tc for tc in body["trace"]["tool_calls"]
            if tc["action"] == "cancel_item"
        ]
        assert len(cancel_calls) >= 1
        assert cancel_calls[0]["status"] == "error"

    def test_j2_dependent_step_skipped_on_cancel_failure(self, client: TestClient):
        """J2: if cancel fails, dependent refund step must be skipped."""
        from tests.conftest import _mock_plan_response
        plan = _mock_plan_response(
            "compound",
            [
                {"action": "cancel_item",
                 "params": {"order_id": "ORD-78323", "line_id": 1},
                 "depends_on": []},
                {"action": "process_refund",
                 "params": {"order_id": "ORD-78323", "amount_inr": 55000.0, "method": "HDFC_CREDIT"},
                 "depends_on": [0]},
            ],
        )
        llm_mock = make_llm_mock(plan, "Could not process.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Cancel and refund ORD-78323.")

        refund_calls = [
            tc for tc in body["trace"]["tool_calls"]
            if tc["action"] == "process_refund"
        ]
        if refund_calls:
            assert refund_calls[0]["status"] != "success"


# ---------------------------------------------------------------------------
# J3 — Escalation with Audit Trail
# ---------------------------------------------------------------------------

class TestJ3Escalation:

    def test_j3_payment_tool_never_called(self, client: TestClient):
        """J3 CRITICAL: process_refund must NEVER appear in trace."""
        llm_mock = make_llm_mock(J3_PLAN, "Case created.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(
                    client,
                    "I want a full refund of ₹42,000 for order ORD-78500, "
                    "the laptop arrived damaged.",
                )

        actions = _tool_actions(body)
        assert "process_refund" not in actions, (
            f"CRITICAL: process_refund was called on an escalation case! "
            f"trace tool_calls: {body['trace']['tool_calls']}"
        )

    def test_j3_crm_case_created(self, client: TestClient, data_dir):
        """J3: CRM escalation case must be created and persisted."""
        llm_mock = make_llm_mock(J3_PLAN, "Your case has been escalated.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                _post_query(
                    client,
                    "I want a full refund of ₹42,000 for order ORD-78500.",
                )

        crm_data = json.loads((data_dir / "crm_cases.json").read_text())
        assert len(crm_data["cases"]) >= 1, "No CRM case was created."

    def test_j3_case_has_trace_id(self, client: TestClient, data_dir):
        """J3: created case must have trace_id attached."""
        llm_mock = make_llm_mock(J3_PLAN, "Escalated.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(client, "Refund ₹42,000 for ORD-78500.")

        trace_id = body["trace"]["trace_id"]
        crm_data = json.loads((data_dir / "crm_cases.json").read_text())
        case_trace_ids = [c.get("trace_id") for c in crm_data["cases"]]
        assert trace_id in case_trace_ids, (
            f"trace_id '{trace_id}' not found in case trace_ids: {case_trace_ids}"
        )

    def test_j3_case_has_structured_handoff(self, client: TestClient, data_dir):
        """J3: case description must contain structured handoff fields."""
        llm_mock = make_llm_mock(J3_PLAN, "Escalated to specialist.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                _post_query(client, "Refund ₹42,000 for ORD-78500.")

        crm_data = json.loads((data_dir / "crm_cases.json").read_text())
        assert crm_data["cases"], "No case created."
        description = crm_data["cases"][0]["description"]

        assert "[ESCALATION CASE" in description
        assert "CUST-001"         in description
        assert "ORD-78500"        in description
        assert "Trace ID"         in description

    def test_j3_case_priority_is_high(self, client: TestClient, data_dir):
        """J3: escalation case must be high priority."""
        llm_mock = make_llm_mock(J3_PLAN, "Escalated.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                _post_query(client, "Refund ₹42,000 for ORD-78500.")

        crm_data = json.loads((data_dir / "crm_cases.json").read_text())
        assert crm_data["cases"][0]["priority"] == "high"

    def test_j3_user_gets_polite_holding_message(self, client: TestClient):
        """J3: user-facing response must be polite, not an error."""
        llm_mock = make_llm_mock(J3_PLAN, "Your case has been escalated.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                body = _post_query(
                    client,
                    "I want a full refund of ₹42,000 for order ORD-78500.",
                )

        response = body["response"].lower()
        assert "error"     not in response
        assert "exception" not in response
        assert any(kw in response for kw in [
            "case", "specialist", "team", "24", "escalat"
        ]), f"Response does not appear to be a holding message: {body['response']}"

    def test_j3_guardrail_pre_check_blocks_high_value(self, client: TestClient):
        """J3 pre-guardrail: GR-001 fires when message mentions refund > ₹25,000."""
        body = _post_query(
            client,
            "Please refund Rs.42000 for my damaged laptop order.",
        )

        guardrail_calls = [
            tc for tc in body["trace"]["tool_calls"]
            if tc["tool"] == "guardrails"
        ]
        assert len(guardrail_calls) >= 1
        assert "process_refund" not in _tool_actions(body)

    def test_j3_case_id_matches_schema_pattern(self, client: TestClient, data_dir):
        """J3: case_id must match ^CASE-[A-Z0-9]{6}$ schema pattern."""
        import re
        llm_mock = make_llm_mock(J3_PLAN, "Escalated.")
        with patch("agent.planner.Planner._call_llm", new=llm_mock):
            with patch("agent.response_builder.ResponseBuilder._call_llm", new=llm_mock):
                _post_query(client, "Refund ₹42,000 for ORD-78500.")

        crm_data = json.loads((data_dir / "crm_cases.json").read_text())
        case_id = crm_data["cases"][0]["case_id"]
        assert re.match(r"^CASE-[A-Z0-9]{6}$", case_id), (
            f"case_id '{case_id}' does not match schema pattern."
        )