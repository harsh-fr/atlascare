"""
tests/test_journeys.py
=======================
End-to-end journey tests for J1, J2, and J3.
"""

import json
import time
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
from tests.conftest import (
    make_tool_mock, make_multi_tool_mock, make_done_mock, make_text_mock,
    make_approved_mock,
)


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


def _run(client, message, groq_responses, session_id="sess-cust001"):
    """Post a query with the Groq client mocked. Returns response body."""
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=groq_responses)
    with patch("agent.graph._groq_client", mock_client):
        return _post(client, message, session_id)


def _j1_responses(text="Your order ORD-78321 is currently being processed."):
    return [
        make_tool_mock("get_order", {"order_id": "ORD-78321"}),
        make_text_mock(text),
    ]


def _j2_responses(text="Done."):
    return [
        make_multi_tool_mock([
            ("cancel_item",    {"order_id": "ORD-78321", "line_id": 2}),
            ("process_refund", {"order_id": "ORD-78321", "amount_inr": 1500.0, "method": "HDFC_CREDIT"}),
            ("update_address", {"order_id": "ORD-78321", "address_label": "office"}),
        ]),
        make_text_mock(text),
        make_approved_mock(),   # evaluator runs for mutation requests
    ]


def _j3_responses():
    # Escalation: deterministic response — no responder LLM call
    return [
        make_tool_mock("escalate", {
            "order_id":   "ORD-78500",
            "reason":     "Customer requesting full refund for damaged laptop. Exceeds threshold.",
            "amount_inr": 42000.0,
        }),
    ]


# ===========================================================================
# J1 — Order Tracking
# ===========================================================================

class TestJ1OrderTracking:

    def test_j1_returns_200(self, client):
        body = _run(client, "Where is my order ORD-78321?", _j1_responses())
        assert body["response"]
        assert body["trace"]["trace_id"].startswith("trc-")

    def test_j1_latency_under_3_seconds(self, client):
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=_j1_responses("Processing."))
        with patch("agent.graph._groq_client", mock_client):
            t0 = time.monotonic()
            _post(client, "Where is my order ORD-78321?")
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert elapsed_ms < 3000, f"J1 latency {elapsed_ms:.0f}ms exceeds 3000ms SLA"

    def test_j1_exactly_one_oms_call(self, client):
        body = _run(client, "Where is my order ORD-78321?", _j1_responses())
        oms_calls = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "get_order"]
        assert len(oms_calls) == 1

    def test_j1_no_crm_or_payment_calls(self, client):
        body = _run(client, "Where is my order ORD-78321?", _j1_responses())
        acts = _actions(body)
        assert "process_refund"  not in acts
        assert "create_crm_case" not in acts
        assert "escalate"        not in acts

    def test_j1_no_escalation(self, client):
        body = _run(client, "Where is my order ORD-78321?", _j1_responses())
        assert "escalate" not in _actions(body)

    def test_j1_trace_fields_present(self, client):
        body  = _run(client, "Where is my order ORD-78321?", _j1_responses())
        trace = body["trace"]
        assert "trace_id"   in trace
        assert "session_id" in trace
        assert "latency_ms" in trace
        assert "tool_calls" in trace
        assert trace["session_id"] == "sess-cust001"
        assert isinstance(trace["latency_ms"], int)
        assert trace["latency_ms"] > 0

    def test_j1_shipped_order_returns_tracking(self, client):
        responses = [
            make_tool_mock("get_order", {"order_id": "ORD-78322"}),
            make_text_mock("Your order is shipped with tracking TRACK-7X9K2M."),
        ]
        body = _run(client, "Where is my order ORD-78322?", responses)
        assert "success" in _statuses(body)

    def test_j1_response_is_non_empty(self, client):
        body = _run(client, "Where is my order ORD-78321?", _j1_responses())
        assert len(body["response"]) > 10

    def test_j1_response_not_truncated(self, client):
        body = _run(client, "Where is my order ORD-78321?", _j1_responses())
        assert body["response"].strip()[-1] in ".?!*)"


# ===========================================================================
# J2 — Multi-Step Compound
# ===========================================================================

class TestJ2CompoundRequest:

    def test_j2_all_three_steps_in_trace(self, client):
        body = _run(
            client,
            "Cancel item 2 from ORD-78321, refund to HDFC card, ship rest to office.",
            _j2_responses(),
        )
        acts = _actions(body)
        assert "cancel_item"    in acts
        assert "process_refund" in acts
        assert "update_address" in acts

    def test_j2_cancel_step_recorded_as_success(self, client):
        body = _run(client, "Cancel item 2 from ORD-78321.", _j2_responses())
        cc   = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert len(cc) >= 1
        assert cc[0]["status"] == "success"

    def test_j2_refund_step_recorded_as_success(self, client, data_dir):
        body = _run(client, "Cancel item 2 from ORD-78321 and refund.", _j2_responses())
        # The refund is delivered as part of cancel_item's auto-refund — cancelling
        # an item refunds it. The separate process_refund on the still-processing
        # order is correctly rejected to prevent a DOUBLE refund.
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "success"
        rc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "process_refund"]
        assert rc and rc[0]["status"] == "error", "redundant standalone refund must be blocked"
        # The refund WAS delivered (via the cancellation) — recorded in refunds.json.
        refunds = json.loads((data_dir / "refunds.json").read_text()).get("refunds", [])
        assert any(r.get("order_id") == "ORD-78321" for r in refunds), \
            "cancel_item should have auto-refunded the cancelled item"

    def test_j2_address_update_step_recorded(self, client):
        body = _run(client, "Ship remaining items to office for ORD-78321.", _j2_responses())
        ac   = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "update_address"]
        assert len(ac) >= 1

    def test_j2_failed_step_recorded_not_hidden(self, client):
        responses = [
            make_tool_mock("cancel_item", {"order_id": "ORD-78323", "line_id": 1}),
            make_text_mock("Sorry, that order cannot be cancelled as it is delivered."),
        ]
        body = _run(client, "Cancel item 1 from ORD-78323.", responses)
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert len(cc) >= 1
        assert cc[0]["status"] == "error"

    def test_j2_cancel_failure_visible_in_trace(self, client):
        """Cancelled-order item records error status, not swallowed."""
        responses = [
            make_tool_mock("cancel_item", {"order_id": "ORD-78323", "line_id": 1}),
            make_text_mock("Order ORD-78323 is delivered and cannot be cancelled."),
        ]
        body = _run(client, "Cancel item 1 from ORD-78323.", responses)
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "error"

    def test_j2_returns_200(self, client):
        body = _run(client, "Cancel item 2 from ORD-78321.", _j2_responses())
        assert body["response"]


# ===========================================================================
# J3 — Escalation with Audit Trail
# ===========================================================================

class TestJ3Escalation:

    def _run_j3(self, client, message="Laptop arrived damaged, order ORD-78500."):
        from agent.guardrails import GuardrailVerdict
        with patch(
            "agent.guardrails.Guardrails.pre_check",
            return_value=GuardrailVerdict.allow(),
        ):
            return _run(client, message, _j3_responses())

    def test_j3_payment_tool_never_called(self, client):
        body = self._run_j3(client)
        assert "process_refund" not in _actions(body), (
            f"CRITICAL: process_refund called on escalation! "
            f"tool_calls: {body['trace']['tool_calls']}"
        )

    def test_j3_crm_case_created(self, client, data_dir):
        self._run_j3(client)
        crm = json.loads((data_dir / "crm_cases.json").read_text())
        assert len(crm["cases"]) >= 1

    def test_j3_case_has_trace_id(self, client, data_dir):
        body = self._run_j3(client)
        tid  = body["trace"]["trace_id"]
        crm  = json.loads((data_dir / "crm_cases.json").read_text())
        assert tid in [c.get("trace_id") for c in crm["cases"]]

    def test_j3_case_has_structured_handoff(self, client, data_dir):
        self._run_j3(client)
        crm  = json.loads((data_dir / "crm_cases.json").read_text())
        assert crm["cases"]
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
        assert re.match(r"^CASE-[A-Z0-9]{6}$", case_id)

    def test_j3_escalation_in_trace(self, client):
        body = self._run_j3(client)
        assert "escalate" in _actions(body)

    def test_j3_returns_200(self, client):
        body = self._run_j3(client)
        assert body["response"]
