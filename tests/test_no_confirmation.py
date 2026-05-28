"""
tests/test_no_confirmation.py
==============================
Tests for Change 1: agent executes actions immediately without asking for
confirmation via plain text. request_confirmation tool is the only sanctioned
mechanism when deliberate confirmation is needed.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from tests.conftest import make_tool_mock, make_text_mock, make_approved_mock


def _run(client, message, groq_responses, session_id="sess-cust001"):
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=groq_responses)
    with patch("agent.graph._groq_client", mock_client):
        r = client.post("/query", json={"message": message, "session_id": session_id})
        assert r.status_code == 200, f"{r.status_code}: {r.text}"
        return r.json()


def _actions(body):
    return [tc["action"] for tc in body["trace"]["tool_calls"]]


# ===========================================================================
# Prompt / static tests
# ===========================================================================

class TestNoConfirmationPrompt:

    def test_agent_system_contains_confirmation_rule(self):
        from agent.graph import _AGENT_SYSTEM
        assert "CONFIRMATION RULE" in _AGENT_SYSTEM

    def test_agent_system_forbids_plain_text_confirmation(self):
        from agent.graph import _AGENT_SYSTEM
        assert "Never generate a plain-text response asking the customer" in _AGENT_SYSTEM

    def test_agent_system_mentions_request_confirmation_tool(self):
        from agent.graph import _AGENT_SYSTEM
        assert "request_confirmation" in _AGENT_SYSTEM

    def test_request_confirmation_in_tools_list(self):
        from agent.graph import TOOLS
        names = [t["function"]["name"] for t in TOOLS]
        assert "request_confirmation" in names

    def test_request_confirmation_has_required_parameters(self):
        from agent.graph import TOOLS
        tool = next(t for t in TOOLS if t["function"]["name"] == "request_confirmation")
        params = tool["function"]["parameters"]["properties"]
        assert "action"               in params
        assert "action_params"        in params
        assert "confirmation_message" in params

    def test_request_confirmation_required_fields(self):
        from agent.graph import TOOLS
        tool = next(t for t in TOOLS if t["function"]["name"] == "request_confirmation")
        required = tool["function"]["parameters"]["required"]
        assert set(required) == {"action", "action_params", "confirmation_message"}

    def test_cancel_item_line_id_has_description(self):
        from agent.graph import TOOLS
        tool = next(t for t in TOOLS if t["function"]["name"] == "cancel_item")
        line_id_prop = tool["function"]["parameters"]["properties"]["line_id"]
        assert "description" in line_id_prop
        assert "get_order" in line_id_prop["description"].lower()

    def test_cancel_item_tool_instructs_get_order_first(self):
        from agent.graph import TOOLS
        tool = next(t for t in TOOLS if t["function"]["name"] == "cancel_item")
        desc = tool["function"]["description"]
        assert "get_order" in desc.lower()


# ===========================================================================
# Behavioural — agent calls tool immediately for low-value items
# ===========================================================================

class TestImmediateExecution:

    def test_low_value_cancel_executes_without_request_confirmation(self, client):
        """Agent cancels ₹800 Wireless Mouse directly — no request_confirmation in trace."""
        body = _run(
            client,
            "Cancel the Wireless Mouse from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("The Wireless Mouse has been cancelled."),
                make_approved_mock(),
            ],
        )
        acts = _actions(body)
        assert "cancel_item"           in acts
        assert "request_confirmation"  not in acts

    def test_standard_cancel_records_success_not_confirmation(self, client):
        """cancel_item appears in trace with success status."""
        body = _run(
            client,
            "Cancel item 3 from ORD-78321.",
            [
                make_tool_mock("cancel_item", {"order_id": "ORD-78321", "line_id": 3}),
                make_text_mock("Cancelled."),
                make_approved_mock(),
            ],
        )
        cc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "cancel_item"]
        assert cc and cc[0]["status"] == "success"

    def test_address_update_executes_immediately(self, client):
        """Address updates always execute immediately — no confirmation."""
        body = _run(
            client,
            "Change shipping address for ORD-78321 to my office.",
            [
                make_tool_mock("update_address", {"order_id": "ORD-78321", "address_label": "office"}),
                make_text_mock("Address updated."),
                make_approved_mock(),
            ],
        )
        acts = _actions(body)
        assert "update_address"       in acts
        assert "request_confirmation" not in acts

    def test_low_value_refund_executes_immediately(self, client):
        """Refunds under ₹5,000 don't need confirmation."""
        body = _run(
            client,
            "Refund ₹800 for my Wireless Mouse on ORD-78321.",
            [
                make_tool_mock("process_refund", {
                    "order_id": "ORD-78321", "amount_inr": 800.0, "method": "original",
                }),
                make_text_mock("Refund of ₹800 initiated."),
                make_approved_mock(),
            ],
        )
        acts = _actions(body)
        assert "process_refund"       in acts
        assert "request_confirmation" not in acts
