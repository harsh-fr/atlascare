"""
tests/test_guardrails.py
=========================
Guardrail unit and integration tests.

Coverage
--------
  GR-001  High-value refund pre-check
    - Exactly at threshold (₹25,000)  → allow
    - One rupee above (₹25,001)       → block
    - Well above (₹42,000)            → block
    - No refund keyword               → allow even if large amount
    - Multiple amounts, one over      → block

  GR-002  Empty message
    - Empty string   → block
    - Whitespace     → block
    - Valid message  → allow

  GR-003  Message too long
    - 2000 chars  → allow
    - 2001 chars  → block

  GR-004  Post-execution: no payment on escalation
    - Both payment + escalation in result → block (CRITICAL)
    - Payment only                        → allow
    - Escalation only                     → allow
    - Neither                             → allow

  Integration
    - Guardrail trigger recorded in trace
    - Blocked request returns 200 with safe message (not 500)
    - Payment tool independently enforces threshold
"""

import pytest

from agent.guardrails import Guardrails, GuardrailVerdict, _extract_amounts
from observability.tracer import Tracer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracer(session_id: str = "sess-test") -> Tracer:
    return Tracer(session_id=session_id)


def _make_execution_summary(
    payment_success: bool = False,
    escalated: bool = False,
) -> list[dict]:
    """Build an execution_summary list matching graph.py format."""
    summary = []
    if payment_success:
        summary.append({
            "tool": "process_refund", "tool_call_id": "call_refund_01",
            "success": True, "data": {"refund": {"status": "initiated"}},
            "error": "", "escalated": False,
        })
    if escalated:
        summary.append({
            "tool": "escalate", "tool_call_id": "call_escalate_01",
            "success": True, "data": {"case_id": "CASE-TEST01", "escalated": True},
            "error": "", "escalated": True,
        })
    return summary


# ---------------------------------------------------------------------------
# Unit: _extract_amounts helper
# ---------------------------------------------------------------------------

class TestExtractAmounts:

    def test_rupee_symbol(self):
        assert 42000.0 in _extract_amounts("I want a refund of ₹42,000")

    def test_rs_dot_format(self):
        assert 25000.0 in _extract_amounts("refund Rs.25000")

    def test_inr_format(self):
        assert 15000.0 in _extract_amounts("INR 15000")

    def test_rupees_word(self):
        assert 5000.0 in _extract_amounts("5000 rupees please")

    def test_multiple_amounts(self):
        amounts = _extract_amounts("₹1000 and ₹42000")
        assert 1000.0  in amounts
        assert 42000.0 in amounts

    def test_no_amounts(self):
        assert _extract_amounts("I need help with my order") == []

    def test_comma_separated(self):
        assert 1000000.0 in _extract_amounts("₹10,00,000")


# ---------------------------------------------------------------------------
# Unit: Guardrails.pre_check
# ---------------------------------------------------------------------------

class TestPreCheck:

    @pytest.fixture
    def guardrails(self):
        return Guardrails()

    # GR-001 ----------------------------------------------------------------

    def test_gr001_exactly_at_threshold_allows(self, guardrails):
        """₹25,000 is the limit — exactly at threshold should be allowed."""
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="I want a refund of ₹25,000 for my order.",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is False

    def test_gr001_one_rupee_above_blocks(self, guardrails):
        """₹25,001 must be blocked."""
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="Please refund Rs.25001 for my damaged item.",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is True
        assert verdict.rule_id == "GR-001"

    def test_gr001_high_value_blocks(self, guardrails):
        """₹42,000 must be blocked."""
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="I want a full refund of ₹42,000 for my laptop.",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is True
        assert verdict.rule_id == "GR-001"

    def test_gr001_large_amount_no_refund_keyword_allows(self, guardrails):
        """Large amount without refund keyword → allow (not a refund request)."""
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="My order total was ₹50,000. Where is it?",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is False

    def test_gr001_multiple_amounts_one_over_blocks(self, guardrails):
        """If any mentioned amount exceeds threshold in a refund request, block."""
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="I want a refund — I paid ₹1000 for the bag and ₹42000 for the laptop.",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is True
        assert verdict.rule_id == "GR-001"

    def test_gr001_user_message_is_polite(self, guardrails):
        """Blocked message must be polite and not expose internals."""
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="Refund ₹42000 please.",
            customer_id="CUST-001",
            tracer=tracer,
        )
        msg = verdict.user_message.lower()
        assert "specialist" in msg or "team" in msg or "review" in msg
        assert "gr-001"    not in msg
        assert "guardrail" not in msg
        assert "threshold" not in msg or "instant-refund" in msg  # policy mention OK

    def test_gr001_tracer_records_trigger(self, guardrails):
        """GR-001 trigger must be recorded in the tracer."""
        tracer = _make_tracer()
        guardrails.pre_check(
            message="Refund ₹42000 for my order.",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert tracer.had_guardrail_trigger()
        events = tracer.get_guardrail_events()
        assert any(e["rule_id"] == "GR-001" for e in events)

    # GR-002 ----------------------------------------------------------------

    def test_gr002_empty_string_blocks(self, guardrails):
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is True
        assert verdict.rule_id == "GR-002"

    def test_gr002_whitespace_only_blocks(self, guardrails):
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="   \t\n  ",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is True
        assert verdict.rule_id == "GR-002"

    def test_gr002_valid_message_allows(self, guardrails):
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="Where is my order?",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is False

    # GR-003 ----------------------------------------------------------------

    def test_gr003_exactly_2000_chars_allows(self, guardrails):
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="a" * 2000,
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is False

    def test_gr003_2001_chars_blocks(self, guardrails, monkeypatch):
        monkeypatch.setenv("MAX_MESSAGE_LENGTH", "2000")
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="a" * 2001,
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.blocked is True
        assert verdict.rule_id == "GR-003"

    def test_gr003_user_message_mentions_limit(self, guardrails, monkeypatch):
        monkeypatch.setenv("MAX_MESSAGE_LENGTH", "2000")
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="x" * 2001,
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert "2,000" in verdict.user_message or "2000" in verdict.user_message

    # Priority order --------------------------------------------------------

    def test_gr002_fires_before_gr001(self, guardrails):
        """Empty message triggers GR-002, not GR-001 (priority order)."""
        tracer = _make_tracer()
        verdict = guardrails.pre_check(
            message="",
            customer_id="CUST-001",
            tracer=tracer,
        )
        assert verdict.rule_id == "GR-002"


# ---------------------------------------------------------------------------
# Unit: Guardrails.post_check
# ---------------------------------------------------------------------------

class TestPostCheck:

    @pytest.fixture
    def guardrails(self):
        return Guardrails()

    def test_gr004_payment_and_escalation_blocks(self, guardrails):
        """CRITICAL: payment success + escalation = GR-004 block."""
        tracer  = _make_tracer()
        result  = _make_execution_summary(payment_success=True, escalated=True)
        verdict = guardrails.post_check(execution_summary=result, tracer=tracer)
        assert verdict.blocked is True
        assert verdict.rule_id == "GR-004"

    def test_gr004_payment_only_allows(self, guardrails):
        """Payment without escalation is fine."""
        tracer  = _make_tracer()
        result  = _make_execution_summary(payment_success=True, escalated=False)
        verdict = guardrails.post_check(execution_summary=result, tracer=tracer)
        assert verdict.blocked is False

    def test_gr004_escalation_only_allows(self, guardrails):
        """Escalation without payment is fine."""
        tracer  = _make_tracer()
        result  = _make_execution_summary(payment_success=False, escalated=True)
        verdict = guardrails.post_check(execution_summary=result, tracer=tracer)
        assert verdict.blocked is False

    def test_gr004_neither_allows(self, guardrails):
        """Neither payment nor escalation → allow."""
        tracer  = _make_tracer()
        result  = _make_execution_summary(payment_success=False, escalated=False)
        verdict = guardrails.post_check(execution_summary=result, tracer=tracer)
        assert verdict.blocked is False

    def test_gr004_tracer_records_critical_trigger(self, guardrails):
        """GR-004 trigger must be recorded as critical in tracer."""
        tracer = _make_tracer()
        result = _make_execution_summary(payment_success=True, escalated=True)
        guardrails.post_check(execution_summary=result, tracer=tracer)
        assert tracer.had_guardrail_trigger()
        events = tracer.get_guardrail_events()
        assert any(e["rule_id"] == "GR-004" for e in events)


# ---------------------------------------------------------------------------
# Unit: GuardrailVerdict
# ---------------------------------------------------------------------------

class TestGuardrailVerdict:

    def test_allow_verdict_not_blocked(self):
        v = GuardrailVerdict.allow()
        assert v.blocked is False
        assert v.user_message == ""
        assert v.rule_id == ""

    def test_block_verdict_is_blocked(self):
        v = GuardrailVerdict.block(
            rule_id="GR-001",
            reason="Test reason",
            user_message="Please wait.",
        )
        assert v.blocked is True
        assert v.rule_id == "GR-001"
        assert v.user_message == "Please wait."

    def test_verdict_is_frozen(self):
        """Verdict must be immutable — FrozenInstanceError on mutation attempt."""
        v = GuardrailVerdict.allow()
        with pytest.raises(Exception):   # FrozenInstanceError
            v.blocked = True


# ---------------------------------------------------------------------------
# Integration: payment tool threshold enforcement
# ---------------------------------------------------------------------------

class TestPaymentToolThreshold:

    def test_payment_tool_blocks_above_25000(self, patched_env):
        """PaymentTool must raise RefundThresholdError for ₹25,001."""
        import asyncio
        from tools.payment_tool import PaymentTool, RefundThresholdError

        tool = PaymentTool()
        with pytest.raises(RefundThresholdError):
            asyncio.get_event_loop().run_until_complete(
                tool.process_refund(
                    order_id="ORD-78321",
                    amount_inr=25001.0,
                    method="HDFC_CREDIT",
                    customer_id="CUST-001",
                )
            )

    def test_payment_tool_allows_at_25000(self, patched_env):
        """PaymentTool must succeed for exactly ₹25,000."""
        import asyncio
        from tools.payment_tool import PaymentTool

        tool = PaymentTool()
        result = asyncio.get_event_loop().run_until_complete(
            tool.process_refund(
                order_id="ORD-78401",
                amount_inr=25000.0,
                method="HDFC_CREDIT",
                customer_id="CUST-001",
            )
        )
        assert result["status"] == "initiated"
        assert result["amount_inr"] == 25000.0

    def test_payment_tool_blocks_below_zero(self, patched_env):
        """PaymentTool must reject zero or negative amounts."""
        import asyncio
        from tools.payment_tool import PaymentTool, InvalidRefundAmountError

        tool = PaymentTool()
        with pytest.raises(InvalidRefundAmountError):
            asyncio.get_event_loop().run_until_complete(
                tool.process_refund(
                    order_id="ORD-78321",
                    amount_inr=0.0,
                    method="HDFC_CREDIT",
                    customer_id="CUST-001",
                )
            )

    def test_payment_tool_blocks_unsupported_method(self, patched_env):
        """PaymentTool must reject unsupported payment methods."""
        import asyncio
        from tools.payment_tool import PaymentTool, InvalidRefundMethodError

        tool = PaymentTool()
        with pytest.raises(InvalidRefundMethodError):
            asyncio.get_event_loop().run_until_complete(
                tool.process_refund(
                    order_id="ORD-78321",
                    amount_inr=1000.0,
                    method="CRYPTO",
                    customer_id="CUST-001",
                )
            )


# ---------------------------------------------------------------------------
# Integration: blocked request via HTTP client
# ---------------------------------------------------------------------------

class TestBlockedRequestHTTP:

    def test_blocked_request_returns_200_not_500(self, client):
        """A guardrail-blocked request must return HTTP 200 with safe message."""
        resp = client.__class__  # type hint only
        response = client.post(
            "/query",
            json={
                "message": "Please refund Rs.42000 for my laptop.",
                "session_id": "sess-cust001",
            },
        )
        assert response.status_code == 200

    def test_blocked_request_has_trace(self, client):
        """A guardrail-blocked request must still return a valid trace."""
        response = client.post(
            "/query",
            json={
                "message": "Refund ₹50,000 for ORD-78500.",
                "session_id": "sess-cust001",
            },
        )
        body = response.json()
        assert "trace"    in body
        assert "trace_id" in body["trace"]
        assert body["trace"]["trace_id"].startswith("trc-")

    def test_blocked_response_is_not_empty(self, client):
        """Blocked response text must be non-empty."""
        response = client.post(
            "/query",
            json={
                "message": "I need a refund of ₹99,000.",
                "session_id": "sess-cust001",
            },
        )
        body = response.json()
        assert body["response"]
        assert len(body["response"]) > 10