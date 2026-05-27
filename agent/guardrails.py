"""
agent/guardrails.py
===================
Deterministic policy enforcement layer.

Responsibility
--------------
  Pre-check  : runs BEFORE the LLM sees the message.
               Catches obvious policy violations early.
  Post-check : runs AFTER tool execution completes.
               Verifies no autonomous payment was made when it
               shouldn't have been (e.g. escalation cases).

Design principles
-----------------
- Every rule here is CODE, not a prompt instruction.
  The LLM cannot override, reinterpret, or bypass these checks.
- Rules are explicit, named, and independently testable.
- GuardrailVerdict is immutable — callers read it, never mutate it.
- The refund threshold (Rs.25,000) is sourced from an environment
  variable with a hardcoded safe default, so it can be changed via
  config without a code deploy.
- Guardrails are intentionally conservative: when in doubt, block.
"""

import logging
import os
import re
from dataclasses import dataclass

from observability.tracer import Tracer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

# Rs.25,000 auto-refund threshold — sourced from env with safe default.
# NEVER change the default without a compliance review.
AUTO_REFUND_LIMIT_INR: float = float(
    os.getenv("AUTO_REFUND_LIMIT_INR", "25000.0")
)

# Regex patterns for amount extraction from free text
# Matches: ₹42,000 | Rs.42000 | Rs 42,000 | INR 42000
_AMOUNT_PATTERNS = [
    re.compile(r"[₹Rs\.]+\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
    re.compile(r"INR\s*([\d,]+(?:\.\d{1,2})?)",       re.IGNORECASE),
    re.compile(r"([\d,]+(?:\.\d{1,2})?)\s*(?:rupees?|inr)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Result contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GuardrailVerdict:
    """
    Immutable result of a guardrail check.

    Attributes
    ----------
    blocked      : True  → Pipeline must halt and return user_message.
                   False → pipeline continues.
    reason       : internal audit label (never shown to user).
    user_message : polite holding message shown to the customer when blocked.
                   Empty string when blocked=False.
    rule_id      : identifier of the rule that triggered (for trace/audit).
    """
    blocked: bool
    reason: str = ""
    user_message: str = ""
    rule_id: str = ""

    @classmethod
    def allow(cls) -> "GuardrailVerdict":
        return cls(blocked=False)

    @classmethod
    def block(
        cls,
        rule_id: str,
        reason: str,
        user_message: str,
    ) -> "GuardrailVerdict":
        return cls(
            blocked=True,
            rule_id=rule_id,
            reason=reason,
            user_message=user_message,
        )


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
class Guardrails:
    """
    Stateless policy enforcement.

    All methods are pure functions of their inputs — no side effects,
    no shared state, safe to call concurrently.
    """

    # ------------------------------------------------------------------
    # Pre-execution check (called before LLM)
    # ------------------------------------------------------------------
    def pre_check(
        self,
        message: str,
        customer_id: str,
        tracer: Tracer,
    ) -> GuardrailVerdict:
        """
        Run all pre-execution rules against the raw customer message.

        Rules are evaluated in priority order; the first triggered rule
        wins and short-circuits the rest.

        Returns
        -------
        GuardrailVerdict — blocked=False means pipeline may continue.
        """
        rules = [
            self._rule_high_value_refund_pre,
            self._rule_empty_message,
            self._rule_message_too_long,
        ]

        for rule in rules:
            verdict = rule(message=message, customer_id=customer_id)
            if verdict.blocked:
                tracer.record_guardrail_trigger(
                    rule_id=verdict.rule_id,
                    phase="pre",
                    reason=verdict.reason,
                )
                logger.warning(
                    "Pre-guardrail triggered | rule=%s | customer=%s | trace=%s",
                    verdict.rule_id,
                    customer_id,
                    tracer.trace_id,
                )
                return verdict

        return GuardrailVerdict.allow()

    # ------------------------------------------------------------------
    # Post-execution check (called after tool execution)
    # ------------------------------------------------------------------
    def post_check(
        self,
        execution_summary: list[dict],
        tracer: Tracer,
    ) -> GuardrailVerdict:
        """
        Verify execution outcomes comply with policy.

        Critical invariant: if an escalation succeeded, no payment
        step should also have succeeded in the same turn.

        Returns
        -------
        GuardrailVerdict — blocked=False means pipeline may continue.
        """
        rules = [self._rule_no_payment_on_escalation]
        for rule in rules:
            verdict = rule(execution_summary)
            if verdict.blocked:
                tracer.record_guardrail_trigger(
                    rule_id=verdict.rule_id,
                    phase="post",
                    reason=verdict.reason,
                )
                logger.error(
                    "Post-guardrail triggered | rule=%s | trace=%s | CRITICAL",
                    verdict.rule_id,
                    tracer.trace_id,
                )
                return verdict
        return GuardrailVerdict.allow()

    # ------------------------------------------------------------------
    # Individual rules — pre
    # ------------------------------------------------------------------
    def _rule_high_value_refund_pre(
        self,
        message: str,
        customer_id: str,
        **_,
    ) -> GuardrailVerdict:
        """
        RULE: GR-001
        If the message mentions a refund amount that exceeds the
        AUTO_REFUND_LIMIT_INR, block autonomous processing immediately
        and route to escalation.

        This rule fires BEFORE the LLM so that the threshold decision
        is never delegated to prompt logic.

        Note: PaymentTool also enforces this via its own
        guard — defence in depth.
        """
        amounts = _extract_amounts(message)
        refund_keywords = re.search(
            r"\b(refund|return|money back|reimburs)\b",
            message,
            re.IGNORECASE,
        )

        if not refund_keywords or not amounts:
            return GuardrailVerdict.allow()

        over_threshold = [a for a in amounts if a > AUTO_REFUND_LIMIT_INR]
        if not over_threshold:
            return GuardrailVerdict.allow()

        max_amount = max(over_threshold)
        return GuardrailVerdict.block(
            rule_id="GR-001",
            reason=(
                f"Refund amount ₹{max_amount:,.2f} exceeds auto-refund "
                f"limit of ₹{AUTO_REFUND_LIMIT_INR:,.0f}. Escalation required."
            ),
            user_message=(
                f"Thank you for reaching out. Your refund request of "
                f"₹{max_amount:,.0f} exceeds our instant-refund limit and "
                "requires review by our specialist team. I'm creating a "
                "priority case for you right now — a specialist will contact "
                "you within 24 hours. We apologise for the inconvenience."
            ),
        )

    def _rule_empty_message(
        self,
        message: str,
        **_,
    ) -> GuardrailVerdict:
        """
        RULE: GR-002
        Reject empty or whitespace-only messages before hitting the LLM.
        """
        if not message or not message.strip():
            return GuardrailVerdict.block(
                rule_id="GR-002",
                reason="Empty message received.",
                user_message=(
                    "It looks like your message was empty. "
                    "Please describe how I can help you today."
                ),
            )
        return GuardrailVerdict.allow()

    def _rule_message_too_long(
        self,
        message: str,
        **_,
    ) -> GuardrailVerdict:
        """
        RULE: GR-003
        Reject messages over 2,000 characters to protect LLM token budget
        and guard against prompt injection via oversized payloads.
        """
        max_len = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
        if len(message) > max_len:
            return GuardrailVerdict.block(
                rule_id="GR-003",
                reason=f"Message length {len(message)} exceeds limit {max_len}.",
                user_message=(
                    "Your message is too long for me to process. "
                    "Please keep your request under 2,000 characters and try again."
                ),
            )
        return GuardrailVerdict.allow()

    # ------------------------------------------------------------------
    # Individual rules — post
    # ------------------------------------------------------------------
    def _rule_no_payment_on_escalation(
        self,
        execution_summary: list[dict],
    ) -> GuardrailVerdict:
        """
        RULE: GR-004 (CRITICAL)
        If a refund was processed AND an escalation case was created in
        the same turn, something has gone wrong — block and alert.

        Defence-in-depth: PaymentTool also checks the threshold, and
        pre-guardrail GR-001 fires before the LLM. This is the final net.
        """
        payment_succeeded = any(
            e["tool"] == "process_refund" and e["success"]
            for e in execution_summary
        )
        escalation_exists = any(
            e["tool"] == "escalate" and e["success"]
            for e in execution_summary
        )

        if payment_succeeded and escalation_exists:
            return GuardrailVerdict.block(
                rule_id="GR-004",
                reason=(
                    "CRITICAL: autonomous refund processed on an escalation case. "
                    "This violates the Rs.25,000 threshold policy."
                ),
                user_message=(
                    "We've detected an issue with your request processing and "
                    "have paused it for safety. Our team has been alerted and "
                    "will reach out to you within 24 hours to resolve this."
                ),
            )
        return GuardrailVerdict.allow()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_amounts(text: str) -> list[float]:
    """
    Extract all monetary amounts from free text.
    Returns a list of floats (may be empty).
    """
    amounts: list[float] = []
    for pattern in _AMOUNT_PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1).replace(",", "")
            try:
                amounts.append(float(raw))
            except ValueError:
                pass
    return amounts