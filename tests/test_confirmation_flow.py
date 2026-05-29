"""
tests/test_confirmation_flow.py
================================
Tests for Change 2: confirmation back-edge.

Architecture summary
--------------------
- `confirmation_check_node` runs as the FIRST node every turn.
- If `awaiting_confirmation` is True in state (persisted by MemorySaver):
    - affirmative ("yes", "ok", …) → dispatches the stored action, routes to
      post_guardrail → responder → evaluator
    - negative ("no", "cancel", …) → clears state, returns canned response, END
    - anything else → clears state, falls through to normal pipeline
- `request_confirmation` tool in TOOLS lets the agent store a pending action
  (set by tool_executor_node when it sees the tool result).
- Multi-turn tests share one `client` fixture instance (same MemorySaver).
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from tests.conftest import (
    make_tool_mock, make_text_mock, make_approved_mock,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _post(client, message, session_id="sess-cust001"):
    r = client.post("/query", json={"message": message, "session_id": session_id})
    assert r.status_code == 200, f"{r.status_code}: {r.text}"
    return r.json()


def _run(client, message, groq_responses, session_id="sess-cust001"):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=groq_responses)
    with patch("agent.graph._groq_client", mock_client):
        return _post(client, message, session_id)


def _actions(body):
    return [tc["action"] for tc in body["trace"]["tool_calls"]]


def _make_request_confirmation_mock(action, action_params, message):
    return make_tool_mock("request_confirmation", {
        "action":               action,
        "action_params":        action_params,
        "confirmation_message": message,
    })


# ===========================================================================
# Unit tests — routing function
# ===========================================================================

class TestRouteConfirmationCheck:

    def test_negative_path_returns_end(self):
        from agent.graph import _route_confirmation_check
        state = {"guardrail_blocked": True, "execution_summary": [], "awaiting_confirmation": False}
        assert _route_confirmation_check(state) == "end"

    def test_affirmative_path_returns_post_guardrail(self):
        from agent.graph import _route_confirmation_check
        state = {
            "guardrail_blocked":     False,
            "awaiting_confirmation": False,
            "execution_summary":     [{"tool": "cancel_item", "success": True}],
        }
        assert _route_confirmation_check(state) == "post_guardrail"

    def test_passthrough_returns_pre_guardrail(self):
        from agent.graph import _route_confirmation_check
        state = {
            "guardrail_blocked":     False,
            "awaiting_confirmation": False,
            "execution_summary":     [],
        }
        assert _route_confirmation_check(state) == "pre_guardrail"

    def test_passthrough_when_awaiting_but_no_action_dispatched(self):
        """awaiting_confirmation cleared + empty execution_summary → pre_guardrail."""
        from agent.graph import _route_confirmation_check
        state = {
            "guardrail_blocked":     False,
            "awaiting_confirmation": False,
            "execution_summary":     [],
        }
        assert _route_confirmation_check(state) == "pre_guardrail"


# ===========================================================================
# Unit tests — regex constants
# ===========================================================================

class TestConfirmationRegex:

    @pytest.mark.parametrize("msg", [
        "yes", "Yes", "YES", "yeah", "yep", "yup", "ok", "okay", "sure",
        "go ahead", "proceed", "do it", "confirm", "absolutely", "fine", "alright",
        "yes!", "ok.", "Sure.",
    ])
    def test_affirmative_matches(self, msg):
        from agent.graph import _AFFIRMATIVE_RE
        assert _AFFIRMATIVE_RE.match(msg), f"Expected affirmative match for: {msg!r}"

    @pytest.mark.parametrize("msg", [
        "no", "No", "NO", "nope", "nah", "cancel", "stop", "don't", "nevermind",
        "never mind", "skip", "abort", "no!", "nope.",
    ])
    def test_negative_matches(self, msg):
        from agent.graph import _NEGATIVE_RE
        assert _NEGATIVE_RE.match(msg), f"Expected negative match for: {msg!r}"

    @pytest.mark.parametrize("msg", [
        "yes but change the address too",
        "no actually cancel order ORD-78321",
        "I want to cancel the laptop",
        "ok show me all my orders",
    ])
    def test_long_messages_do_not_match_affirmative(self, msg):
        from agent.graph import _AFFIRMATIVE_RE
        assert not _AFFIRMATIVE_RE.match(msg)

    @pytest.mark.parametrize("msg", [
        "no but refund the other item",
        "cancel order ORD-78321 instead",
    ])
    def test_long_messages_do_not_match_negative(self, msg):
        from agent.graph import _NEGATIVE_RE
        assert not _NEGATIVE_RE.match(msg)


# ===========================================================================
# Tool / dispatch tests
# ===========================================================================

class TestRequestConfirmationTool:

    def test_request_confirmation_in_tools_list(self):
        from agent.graph import TOOLS
        names = [t["function"]["name"] for t in TOOLS]
        assert "request_confirmation" in names

    def test_request_confirmation_stores_pending_action(self, client):
        """When model calls request_confirmation, response contains the confirmation prompt."""
        body = _run(
            client,
            "Cancel the Dell laptop from ORD-78321.",
            [
                _make_request_confirmation_mock(
                    "cancel_item",
                    {"order_id": "ORD-78321", "line_id": 1},
                    "The Dell Inspiron laptop costs ₹55,000. Are you sure you want to cancel it?",
                ),
                make_text_mock("The Dell Inspiron laptop costs ₹55,000. Are you sure you want to cancel it?"),
                # evaluator bypasses — request_confirmation in tools_called
            ],
        )
        assert body["response"]
        assert "request_confirmation" in _actions(body)

    def test_request_confirmation_trace_shows_success(self, client):
        body = _run(
            client,
            "Cancel the Dell laptop from ORD-78321.",
            [
                _make_request_confirmation_mock(
                    "cancel_item",
                    {"order_id": "ORD-78321", "line_id": 1},
                    "This laptop costs ₹55,000. Confirm?",
                ),
                make_text_mock("This laptop costs ₹55,000. Confirm?"),
            ],
        )
        rc = [tc for tc in body["trace"]["tool_calls"]
              if tc["action"] == "request_confirmation"]
        assert rc and rc[0]["status"] == "success"

    def test_unsupported_method_in_confirmation_is_refused(self, client):
        """The model must NOT be able to gate an unsupported refund method behind a
        confirmation prompt. The handler refuses to stage it and surfaces the menu —
        so the bot never asks 'are you sure?' for a method it can't refund to.
        Neutral message + a non-hint method ('gift card') so the deterministic
        message-level guard does not pre-empt the confirmation-handler path."""
        body = _run(
            client,
            "Cancel item 1 from ORD-78321.",
            [
                _make_request_confirmation_mock(
                    "cancel_item",
                    {"order_id": "ORD-78321", "line_id": 1, "refund_method": "gift card"},
                    "Cancel the item and refund to your gift card — are you sure?",
                ),
                make_text_mock("That method isn't supported — pick your original method or a supported one."),
                make_approved_mock(),
            ],
        )
        rc = [tc for tc in body["trace"]["tool_calls"]
              if tc["action"] == "request_confirmation"]
        assert rc and rc[0]["status"] == "error", \
            "an unsupported refund method must not be staged for confirmation"

    def test_low_value_cancel_skips_confirmation(self, client):
        """Anti-overcaution: a confirmation staged for a low-value (≤₹5,000), concrete
        cancel is executed directly as cancel_item — the bot does NOT ask 'are you
        sure?'. (ORD-78321 line 3 is an ₹800 mouse.)"""
        body = _run(
            client,
            "Cancel the wireless mouse on ORD-78321.",
            [
                _make_request_confirmation_mock(
                    "cancel_item",
                    {"order_id": "ORD-78321", "line_id": 3},
                    "The Wireless Mouse costs ₹800. Are you sure you want to cancel it?",
                ),
                make_text_mock("Your Wireless Mouse has been cancelled and a refund initiated."),
                make_approved_mock(),
            ],
        )
        acts = _actions(body)
        assert "cancel_item" in acts, "low-value cancel should execute directly"
        assert "request_confirmation" not in acts, "low-value cancel must not be gated by confirmation"
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "success"

    def test_high_value_cancel_still_confirms(self, client):
        """Guard the other side: a >₹5,000 item (line 1 = ₹55,000 laptop) is STILL
        gated behind a confirmation — the bypass must not weaken that."""
        body = _run(
            client,
            "Cancel the Dell laptop on ORD-78321.",
            [
                _make_request_confirmation_mock(
                    "cancel_item",
                    {"order_id": "ORD-78321", "line_id": 1},
                    "The Dell laptop costs ₹55,000. Are you sure?",
                ),
                make_text_mock("The Dell laptop costs ₹55,000. Are you sure?"),
            ],
        )
        assert "request_confirmation" in _actions(body)
        assert "cancel_item" not in _actions(body), "high-value must wait for explicit confirmation"


# ===========================================================================
# Multi-turn integration tests
# ===========================================================================

class TestConfirmationBackEdge:

    def test_affirmative_yes_dispatches_stored_cancel(self, client):
        """
        Turn 1: model calls request_confirmation → stores cancel_item as pending.
        Turn 2: user says 'yes' → confirmation_check_node dispatches cancel_item.
        """
        SESSION = "sess-cust001"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            # Turn 1 LLM calls: plan + respond (evaluator bypassed for request_confirmation)
            _make_request_confirmation_mock(
                "cancel_item",
                {"order_id": "ORD-78321", "line_id": 1},
                "The Dell Inspiron Laptop costs ₹55,000. Are you sure you want to cancel it?",
            ),
            make_text_mock("The Dell Inspiron Laptop costs ₹55,000. Are you sure?"),
            # Turn 2 LLM calls: respond + eval (cancel_item is a mutation)
            make_text_mock("Your Dell Inspiron Laptop has been cancelled. A refund of ₹55,000 has been initiated."),
            make_approved_mock(),
        ])

        with patch("agent.graph._groq_client", mock_client):
            _post(client, "Cancel the Dell laptop from ORD-78321.", SESSION)
            body2 = _post(client, "yes", SESSION)

        acts = _actions(body2)
        assert "cancel_item" in acts

    def test_affirmative_ok_dispatches_stored_cancel(self, client):
        """'ok' is treated as affirmative."""
        SESSION = "sess-cust001"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _make_request_confirmation_mock(
                "cancel_item",
                {"order_id": "ORD-78321", "line_id": 1},
                "₹55,000 item. Confirm cancellation?",
            ),
            make_text_mock("₹55,000 item. Confirm cancellation?"),
            make_text_mock("Cancelled. Refund initiated."),
            make_approved_mock(),
        ])

        with patch("agent.graph._groq_client", mock_client):
            _post(client, "Cancel the laptop from ORD-78321.", SESSION)
            body2 = _post(client, "ok", SESSION)

        assert "cancel_item" in _actions(body2)

    def test_affirmative_dispatch_status_is_success(self, client):
        """cancel_item dispatched from confirmation_check_node records success."""
        SESSION = "sess-cust001"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _make_request_confirmation_mock(
                "cancel_item",
                {"order_id": "ORD-78321", "line_id": 1},
                "Confirm cancel?",
            ),
            make_text_mock("Confirm cancel?"),
            make_text_mock("Cancelled."),
            make_approved_mock(),
        ])

        with patch("agent.graph._groq_client", mock_client):
            _post(client, "Cancel laptop from ORD-78321.", SESSION)
            body2 = _post(client, "yes", SESSION)

        cc = [tc for tc in body2["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "success"

    def test_negative_no_clears_pending_no_cancel(self, client):
        """
        Turn 1: model calls request_confirmation.
        Turn 2: user says 'no' → cancel_item NOT dispatched, canned response returned.
        """
        SESSION = "sess-cust001"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _make_request_confirmation_mock(
                "cancel_item",
                {"order_id": "ORD-78321", "line_id": 1},
                "Confirm?",
            ),
            make_text_mock("This ₹55,000 laptop. Confirm cancel?"),
        ])

        with patch("agent.graph._groq_client", mock_client):
            _post(client, "Cancel laptop from ORD-78321.", SESSION)
            body2 = _post(client, "no", SESSION)

        assert "cancel_item"      not in _actions(body2)
        assert "cancelled"        in body2["response"].lower() or \
               "anything else"    in body2["response"].lower()

    def test_negative_cancel_clears_pending(self, client):
        """'cancel' as a reply is treated as negative."""
        SESSION = "sess-cust001"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _make_request_confirmation_mock(
                "cancel_item",
                {"order_id": "ORD-78321", "line_id": 1},
                "Confirm?",
            ),
            make_text_mock("Confirm cancel?"),
        ])

        with patch("agent.graph._groq_client", mock_client):
            _post(client, "Cancel laptop from ORD-78321.", SESSION)
            body2 = _post(client, "cancel", SESSION)

        assert "cancel_item" not in _actions(body2)

    def test_negative_response_returns_200(self, client):
        """Negative path ends cleanly — no errors."""
        SESSION = "sess-cust001"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            _make_request_confirmation_mock(
                "cancel_item", {"order_id": "ORD-78321", "line_id": 1}, "Confirm?",
            ),
            make_text_mock("Confirm?"),
        ])

        with patch("agent.graph._groq_client", mock_client):
            _post(client, "Cancel laptop from ORD-78321.", SESSION)
            r = client.post("/query", json={"message": "no", "session_id": SESSION})

        assert r.status_code == 200
        assert r.json()["response"]

    def test_topic_change_clears_pending_and_processes_new_request(self, client):
        """
        Turn 1: request_confirmation stores pending action.
        Turn 2: user asks about orders (topic change) → pending cleared,
                list_orders called, no cancel_item.
        """
        SESSION = "sess-cust001"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            # Turn 1
            _make_request_confirmation_mock(
                "cancel_item", {"order_id": "ORD-78321", "line_id": 1}, "Confirm?",
            ),
            make_text_mock("Confirm?"),
            # Turn 2 — new request processed normally
            make_tool_mock("list_orders", {}),
            make_text_mock("Here are all your orders."),
        ])

        with patch("agent.graph._groq_client", mock_client):
            _post(client, "Cancel laptop from ORD-78321.", SESSION)
            body2 = _post(client, "Show me all my orders.", SESSION)

        assert "cancel_item"  not in _actions(body2)
        assert "list_orders"  in _actions(body2)

    def test_after_negative_next_turn_works_normally(self, client):
        """After a 'no', the following turn should process normally."""
        SESSION = "sess-cust001"
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=[
            # Turn 1: confirmation request
            _make_request_confirmation_mock(
                "cancel_item", {"order_id": "ORD-78321", "line_id": 1}, "Confirm?",
            ),
            make_text_mock("Confirm?"),
            # Turn 2: user says no
            # (no LLM call — confirmation_check_node handles it deterministically)
            # Turn 3: normal order lookup
            make_tool_mock("get_order", {"order_id": "ORD-78321"}),
            make_text_mock("Your order is processing."),
        ])

        with patch("agent.graph._groq_client", mock_client):
            _post(client, "Cancel laptop from ORD-78321.", SESSION)
            _post(client, "no", SESSION)
            body3 = _post(client, "Where is my order ORD-78321?", SESSION)

        assert "get_order" in _actions(body3)

    def test_low_value_item_cancels_immediately_no_confirmation(self, client):
        """Items under ₹5,000 threshold cancel directly, no request_confirmation."""
        body = _run(
            client,
            "Cancel the Wireless Mouse from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("Wireless Mouse cancelled. Refund of ₹800 initiated."),
                make_approved_mock(),
            ],
            session_id="sess-cust001",
        )
        assert "cancel_item"           in _actions(body)
        assert "request_confirmation"  not in _actions(body)
