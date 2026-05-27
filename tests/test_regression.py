"""
tests/test_regression.py
=========================
Regression tests covering all recent fixes.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from tests.conftest import make_tool_mock, make_done_mock, make_text_mock


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


def _j1_responses(text="Your order is being processed."):
    return [
        make_tool_mock("get_order", {"order_id": "ORD-78321"}),
        make_text_mock(text),
    ]


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
        r = resp.json()["response"].lower()
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
        from agent.graph import _normalise_refund_method
        result = _normalise_refund_method(raw)
        assert result == expected

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
        body = _run(client, "Where is ORD-00000?", [
            make_tool_mock("get_order", {"order_id": "ORD-00000"}),
            make_text_mock("I'm sorry, I couldn't find order ORD-00000 in your account."),
        ])
        resp = body["response"]
        assert "ORD-00000" in resp or "not find" in resp.lower() or "not found" in resp.lower()

    def test_not_found_does_not_say_system_error(self, client):
        body = _run(client, "Where is ORD-00000?", [
            make_tool_mock("get_order", {"order_id": "ORD-00000"}),
            make_text_mock("I couldn't find that order."),
        ])
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
        r = resp.json()["response"].lower()
        assert any(k in r for k in ["format", "ord-", "xxxxx", "example", "5 digit"])

    def test_valid_order_id_not_caught_as_invalid(self, client):
        body = _run(client, "Track ORD-78321", _j1_responses())
        r    = body["response"].lower()
        assert "format" not in r
        assert "xxxxx"  not in r


# ===========================================================================
# Order tracking response quality
# ===========================================================================

class TestOrderTrackingResponse:

    def test_order_tracking_response_has_order_details(self, client):
        body = _run(client, "Where is ORD-78321?",
                    _j1_responses("Your order ORD-78321 is currently processing."))
        resp = body["response"]
        assert "ORD-78321" in resp
        assert any(w in resp.lower() for w in ["processing", "shipped", "delivered", "placed", "status"])

    def test_order_tracking_response_not_truncated(self, client):
        body = _run(client, "Where is ORD-78321?", _j1_responses("Complete."))
        resp = body["response"].strip()
        assert resp[-1] in ".?!*)"


# ===========================================================================
# Vague help detection
# ===========================================================================

class TestVagueHelpDetection:

    @pytest.mark.parametrize("msg", [
        "help",
        "i need help",
        "need help with my order",
        "can you help me",
    ])
    def test_vague_message_returns_graceful_prompt(self, client, msg):
        responses = [
            make_done_mock("I need your order ID to help you."),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)
        with patch("agent.graph._groq_client", mock_client):
            resp = client.post("/query", json={"message": msg, "session_id": "sess-cust001"})
        assert resp.status_code == 200
        body = resp.json()["response"].lower()
        assert "order" in body or "ord-" in body or "id" in body

    @pytest.mark.parametrize("msg", ["hi", "Hello", "hey"])
    def test_greeting_returns_welcome_response(self, client, msg):
        responses = [
            make_done_mock("Hello! Welcome to AtlasCare."),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)
        with patch("agent.graph._groq_client", mock_client):
            resp = client.post("/query", json={"message": msg, "session_id": "sess-cust001"})
        assert resp.status_code == 200
        body = resp.json()["response"].lower()
        assert "doesn't look" not in body
        assert "invalid"      not in body

    def test_vague_help_does_not_say_invalid_format(self, client):
        responses = [
            make_done_mock("I need your order ID."),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)
        with patch("agent.graph._groq_client", mock_client):
            resp = client.post("/query",
                json={"message": "need help", "session_id": "sess-cust001"})
        body = resp.json()["response"].lower()
        assert "doesn't look quite right" not in body
        assert "doesn't look"             not in body
        assert "error"                    not in body


# ===========================================================================
# Refund with explicit amount
# ===========================================================================

class TestRefundWithAmount:

    def test_refund_explicit_amount_succeeds(self, client):
        body = _run(client, "I want a refund of Rs.24999 for order ORD-78400.", [
            make_tool_mock("process_refund", {
                "order_id": "ORD-78400", "amount_inr": 24999.0, "method": "original",
            }),
            make_text_mock("Your refund of ₹24,999 has been initiated."),
        ])
        rc = [tc for tc in body["trace"]["tool_calls"] if tc["action"] == "process_refund"]
        assert rc and rc[0]["status"] == "success"
