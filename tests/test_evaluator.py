"""
tests/test_evaluator.py
========================
Tests for Change 3: response evaluator loop.

Bypass conditions (no LLM call):
  - eval_retry_count >= 2
  - tool_call_count == 0 or execution_summary empty
  - escalation in execution_summary
  - "request_confirmation" in tools_called
  - tools_called ⊆ _READ_ONLY_TOOLS and not _is_complex

When the evaluator runs:
  - APPROVED  → route to END
  - REJECTED  → route back to responder with eval_feedback injected in prompt
  - 2nd REJECTED → circuit break, eval_approved = True, route to END
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from tests.conftest import (
    make_tool_mock, make_multi_tool_mock, make_text_mock,
    make_approved_mock, make_rejected_mock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(client, message, groq_responses, session_id="sess-cust001"):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=groq_responses)
    with patch("agent.graph._groq_client", mock_client):
        r = client.post("/query", json={"message": message, "session_id": session_id})
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        return r.json()


def _tools_in_trace(body):
    return [tc["tool"] for tc in body["trace"]["tool_calls"]]


def _actions(body):
    return [tc["action"] for tc in body["trace"]["tool_calls"]]


def _count_tool_in_trace(body, tool_name):
    return sum(1 for tc in body["trace"]["tool_calls"] if tc["tool"] == tool_name)


# ===========================================================================
# Unit tests — routing function
# ===========================================================================

class TestRouteEvaluator:

    def test_approved_routes_to_end(self):
        from agent.graph import _route_evaluator
        assert _route_evaluator({"eval_approved": True,  "eval_retry_count": 0}) == "end"

    def test_max_retries_routes_to_end(self):
        from agent.graph import _route_evaluator
        assert _route_evaluator({"eval_approved": False, "eval_retry_count": 2}) == "end"

    def test_rejected_first_time_routes_to_responder(self):
        from agent.graph import _route_evaluator
        assert _route_evaluator({"eval_approved": False, "eval_retry_count": 1}) == "responder"

    def test_rejected_zero_retries_routes_to_responder(self):
        from agent.graph import _route_evaluator
        assert _route_evaluator({"eval_approved": False, "eval_retry_count": 0}) == "responder"

    def test_approved_with_retries_still_routes_to_end(self):
        from agent.graph import _route_evaluator
        assert _route_evaluator({"eval_approved": True, "eval_retry_count": 1}) == "end"


# ===========================================================================
# Bypass — no extra LLM call
# ===========================================================================

class TestEvaluatorBypass:

    def test_j1_order_lookup_does_not_call_evaluator_llm(self, client):
        """get_order only (read-only, not complex) → evaluator bypasses. 2 LLM calls total."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            make_tool_mock("get_order", {"order_id": "ORD-78321"}),
            make_text_mock("Your order is processing."),
            # No 3rd call — evaluator bypasses
        ])
        with patch("agent.graph._groq_client", mock_client):
            r = client.post("/query", json={
                "message": "Where is my order ORD-78321?", "session_id": "sess-cust001"
            })
        assert r.status_code == 200
        # If evaluator had called LLM, side_effect would be exhausted and the
        # call would raise StopAsyncIteration → 500 response. 200 = bypass confirmed.

    def test_list_orders_does_not_call_evaluator_llm(self, client):
        """list_orders is read-only — evaluator bypasses. 2 LLM calls."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            make_tool_mock("list_orders", {}),
            make_text_mock("Here are your orders."),
        ])
        with patch("agent.graph._groq_client", mock_client):
            r = client.post("/query", json={
                "message": "Show me all my orders.", "session_id": "sess-cust001"
            })
        assert r.status_code == 200

    def test_direct_text_answer_no_tools_bypasses_evaluator(self, client):
        """No tool calls → evaluator bypasses. 1 LLM call."""
        from tests.conftest import make_done_mock
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            make_done_mock("Hello! How can I help you today?"),
        ])
        with patch("agent.graph._groq_client", mock_client):
            r = client.post("/query", json={
                "message": "hi", "session_id": "sess-cust001"
            })
        assert r.status_code == 200

    def test_j1_latency_with_evaluator_bypass(self, client):
        """Evaluator bypass preserves J1 latency SLA (<3 s)."""
        import time
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            make_tool_mock("get_order", {"order_id": "ORD-78321"}),
            make_text_mock("Processing."),
        ])
        with patch("agent.graph._groq_client", mock_client):
            t0 = time.monotonic()
            r  = client.post("/query", json={
                "message": "Where is my order ORD-78321?", "session_id": "sess-cust001"
            })
            elapsed_ms = (time.monotonic() - t0) * 1000
        assert r.status_code == 200
        assert elapsed_ms < 3000, f"Latency {elapsed_ms:.0f}ms with evaluator bypass"

    def test_escalation_bypasses_evaluator(self, client):
        """Escalation uses a deterministic response — evaluator bypasses. 1 LLM call."""
        from agent.guardrails import GuardrailVerdict
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            make_tool_mock("escalate", {
                "order_id": "ORD-78500",
                "reason":   "Customer reports damaged product.",
                "amount_inr": 42000.0,
            }),
            # No 2nd call — evaluator bypasses for escalation
        ])
        with patch("agent.graph._groq_client", mock_client):
            with patch(
                "agent.guardrails.Guardrails.pre_check",
                return_value=GuardrailVerdict.allow(),
            ):
                r = client.post("/query", json={
                    "message": "Laptop arrived damaged, order ORD-78500.",
                    "session_id": "sess-cust001",
                })
        assert r.status_code == 200

    def test_request_confirmation_tool_bypasses_evaluator(self, client):
        """Confirmation prompts need no evaluation. 2 LLM calls: plan + respond."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            make_tool_mock("request_confirmation", {
                "action":               "cancel_item",
                "action_params":        {"order_id": "ORD-78321", "line_id": 1},
                "confirmation_message": "This is a ₹55,000 item. Confirm?",
            }),
            make_text_mock("This is a ₹55,000 item. Are you sure?"),
        ])
        with patch("agent.graph._groq_client", mock_client):
            r = client.post("/query", json={
                "message": "Cancel the laptop from ORD-78321.", "session_id": "sess-cust001"
            })
        assert r.status_code == 200


# ===========================================================================
# Evaluator runs and approves
# ===========================================================================

class TestEvaluatorApproval:

    def test_evaluator_runs_for_cancel_item_mutation(self, client):
        """cancel_item is a mutation → evaluator calls LLM. 3 LLM calls: plan + respond + eval."""
        body = _run(
            client,
            "Cancel item 3 from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("Wireless Mouse cancelled. Refund of ₹800 initiated."),
                make_approved_mock(),
            ],
        )
        assert body["response"]
        assert body["task_complete"]

    def test_evaluator_runs_for_process_refund(self, client):
        """process_refund → evaluator calls LLM."""
        body = _run(
            client,
            "Refund ₹800 for order ORD-78321.",
            [
                make_tool_mock("process_refund", {
                    "order_id": "ORD-78321", "amount_inr": 800.0, "method": "original",
                }),
                make_text_mock("Refund of ₹800 initiated."),
                make_approved_mock(),
            ],
        )
        assert body["task_complete"]

    def test_evaluator_approved_task_complete_true(self, client):
        body = _run(
            client,
            "Cancel item 3 from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("Cancelled."),
                make_approved_mock(),
            ],
        )
        assert body["task_complete"] is True

    def test_evaluator_approved_response_preserved(self, client):
        """When approved on first try, the original response text is returned unchanged."""
        expected = "Your Wireless Mouse has been cancelled. Refund of ₹800 initiated."
        body = _run(
            client,
            "Cancel item 3 from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock(expected),
                make_approved_mock(),
            ],
        )
        assert body["response"] == expected


# ===========================================================================
# Evaluator rejects — responder retries
# ===========================================================================

class TestEvaluatorRetry:

    def test_evaluator_rejection_triggers_responder_retry(self, client):
        """
        REJECTED on first eval → responder runs again → APPROVED on second eval.
        LLM call sequence: plan + respond(1) + eval(rejected) + respond(2) + eval(approved)
        = 5 LLM calls.
        """
        improved = "Your Wireless Mouse (line_id 3) has been cancelled. A refund of ₹800 has been initiated to your HDFC Credit Card within 5-7 business days."
        body = _run(
            client,
            "Cancel item 3 from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("Cancelled."),                               # respond round 1
                make_rejected_mock("Response omitted refund amount and timeline."),  # eval
                make_text_mock(improved),                                   # respond round 2
                make_approved_mock(),                                        # eval round 2
            ],
        )
        assert body["response"] == improved

    def test_evaluator_retry_returns_improved_response(self, client):
        improved = "Cancelled with ₹800 refund in 5-7 business days."
        body = _run(
            client,
            "Cancel item 3 from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("Done."),
                make_rejected_mock("Missing refund amount."),
                make_text_mock(improved),
                make_approved_mock(),
            ],
        )
        assert body["response"] == improved

    def test_evaluator_circuit_breaks_after_two_rejections(self, client):
        """
        Two REJECTED verdicts → circuit break at eval_retry_count == 2.
        eval_approved forced True, last response accepted.
        LLM calls: plan + respond(1) + eval(reject1) + respond(2) + eval(reject2)
        = 5 LLM calls. No 6th LLM call.
        """
        fallback = "Something went wrong but I'll help further."
        body = _run(
            client,
            "Cancel item 3 from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("Cancelled."),
                make_rejected_mock("Missing refund amount."),
                make_text_mock(fallback),
                make_rejected_mock("Still missing timeline."),
                # No 6th call — circuit break
            ],
        )
        assert body["response"] == fallback

    def test_circuit_break_returns_200(self, client):
        """After two rejections, request still completes with 200."""
        body = _run(
            client,
            "Cancel item 3 from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("Cancelled."),
                make_rejected_mock("Missing amount."),
                make_text_mock("Cancelled with ₹800 refund."),
                make_rejected_mock("Missing timeline."),
            ],
        )
        assert body["response"]

    def test_multi_step_mutation_evaluator_runs(self, client):
        """J2-style multi-tool mutation request triggers evaluator."""
        body = _run(
            client,
            "Cancel item 2 from ORD-78321, refund to HDFC card, ship rest to office.",
            [
                make_multi_tool_mock([
                    ("cancel_item",    {"order_id": "ORD-78321", "line_id": 2}),
                    ("process_refund", {"order_id": "ORD-78321", "amount_inr": 1500.0, "method": "HDFC_CREDIT"}),
                    ("update_address", {"order_id": "ORD-78321", "address_label": "office"}),
                ]),
                make_text_mock("All done."),
                make_approved_mock(),
            ],
        )
        assert body["task_complete"]


# ===========================================================================
# Evaluator constant / configuration tests
# ===========================================================================

class TestEvaluatorConfig:

    def test_read_only_tools_set_contents(self):
        from agent.graph import _READ_ONLY_TOOLS
        assert "get_order"             in _READ_ONLY_TOOLS
        assert "list_orders"           in _READ_ONLY_TOOLS
        assert "list_cases"            in _READ_ONLY_TOOLS
        assert "search_kb"             in _READ_ONLY_TOOLS
        assert "request_confirmation"  in _READ_ONLY_TOOLS

    def test_cancel_item_not_in_read_only_tools(self):
        from agent.graph import _READ_ONLY_TOOLS
        assert "cancel_item"    not in _READ_ONLY_TOOLS
        assert "process_refund" not in _READ_ONLY_TOOLS
        assert "escalate"       not in _READ_ONLY_TOOLS
        assert "update_address" not in _READ_ONLY_TOOLS
