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
from utils.payment_methods import DEFAULT_AUTO_REFUND_LIMIT_INR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------

# Rs.25,000 auto-refund threshold — single source of truth.
def _resolve_auto_refund_limit() -> float:
    """Resolve the auto-refund threshold so every enforcement layer agrees.

    Precedence (mirrors PaymentTool._enforce_threshold):
      1. ``auto_refund_limit_inr`` in payment_config.json — the data the
         operator/evaluator supplies. This is what lets the threshold track
         the deployed config WITHOUT a code change, so a swapped-in data
         folder is honoured by the guardrail and the agent prompt alike.
      2. ``AUTO_REFUND_LIMIT_INR`` env var — fallback when config is missing.
      3. Rs.25,000 hardcoded safe default.

    NEVER change the default without a compliance review.
    """
    try:
        from repositories.payment_repository import PaymentRepository

        limit = PaymentRepository().get_config().get("auto_refund_limit_inr")
        if limit is not None:
            return float(limit)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("auto_refund_limit_inr unavailable from payment_config: %s", exc)
    return float(os.getenv("AUTO_REFUND_LIMIT_INR", str(DEFAULT_AUTO_REFUND_LIMIT_INR)))


AUTO_REFUND_LIMIT_INR: float = _resolve_auto_refund_limit()

# Regex patterns for amount extraction from free text
# Matches: ₹42,000 | Rs.42000 | Rs 42,000 | INR 42000
_AMOUNT_PATTERNS = [
    # ₹42,000 | Rs.42000 | Rs 42,000 — proper alternation (the old
    # character-class form [₹Rs\.]+ matched stray 'R'/'s'/'.' and mis-extracted).
    # \brs avoids matching the "rs" inside words like "hours".
    re.compile(r"(?:₹|\brs\.?)\s*([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE),
    re.compile(r"INR\s*([\d,]+(?:\.\d{1,2})?)",       re.IGNORECASE),
    re.compile(r"([\d,]+(?:\.\d{1,2})?)\s*(?:rupees?|inr)", re.IGNORECASE),
]

# A bare amount with NO currency marker, but anchored to a refund verb, e.g.
# "refund 50000", "return me 30,000". Anchoring to the verb is what makes this
# safe: it cannot pick up an order-id digit ("ORD-10014") or a pincode elsewhere
# in the message. Requires ≥4 digits (or comma-grouped), which covers every
# over-threshold amount while ignoring small/3-digit numbers.
_REFUND_CONTEXT_AMOUNT_RE = re.compile(
    r"\b(?:refund|reimburse|return|credit|money\s*back|pay\s*back)\b"
    r"(?:\s+(?:me|us|of|for|the|a|an|my))*"
    r"\s+(?:₹|rs\.?|inr)?\s*"
    r"(\d{1,3}(?:,\d{3})+|\d{4,})(?:\.\d{1,2})?",
    re.IGNORECASE,
)

# F-08: worded-amount lookup table for common Indian denominations
# Maps normalised word tokens → float multiplier
_WORD_AMOUNT_UNITS: dict[str, float] = {
    "hundred": 100.0,
    "thousand": 1_000.0,
    "lakh": 100_000.0,
    "lac": 100_000.0,
    "crore": 10_000_000.0,
}
_WORD_DIGITS: dict[str, float] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
    "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
}

def _parse_worded_amount(text: str) -> list[float]:
    """
    Extract monetary amounts expressed in words, e.g.
    'twenty-five thousand rupees'  → [25000.0]
    'one lakh'                     → [100000.0]
    Returns a list of floats (may be empty).
    Only fires when a currency word is present nearby.
    """
    lower = text.lower()
    # Only attempt if a currency indicator is nearby
    if not re.search(r'\b(rupees?|inr|rs\.?|₹)\b', lower):
        return []

    amounts: list[float] = []
    # Tokenise: keep hyphens so "twenty-five" works
    tokens = re.findall(r'[a-z]+', lower)
    i = 0
    while i < len(tokens):
        if tokens[i] in _WORD_DIGITS or tokens[i] in _WORD_AMOUNT_UNITS:
            total = 0.0
            current = 0.0
            j = i
            while j < len(tokens):
                t = tokens[j]
                if t in _WORD_DIGITS:
                    current += _WORD_DIGITS[t]
                elif t in _WORD_AMOUNT_UNITS:
                    if current == 0:
                        current = 1
                    current *= _WORD_AMOUNT_UNITS[t]
                    total += current
                    current = 0.0
                elif t in ("and", "rupees", "rupee", "inr", "rs"):
                    j += 1
                    continue
                else:
                    break
                j += 1
            if current > 0:
                total += current
            if total > 0:
                amounts.append(total)
            i = j
        else:
            i += 1
    return amounts


# ---------------------------------------------------------------------------
# High-severity safety / fraud / legal signals (code-enforced escalation)
# ---------------------------------------------------------------------------
# These must be handled by a specialist and must NEVER be resolved by an
# autonomous refund/cancel. This is a CODE backstop for the escalation policy
# that otherwise lives only in the agent prompt (and is bypassable by prompt
# injection or model error). Intentionally conservative: when in doubt, escalate.
_SAFETY_ESCALATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("fraud_or_unauthorised", re.compile(
        r"\b(fraud(?:ulent)?|unauthor(?:ised|ized)"
        r"|did(?:n['’]?t| not)\s+(?:place|order|authori[sz]e)"
        r"|never\s+(?:placed|ordered|authori[sz]ed)|not\s+my\s+order"
        r"|without\s+my\s+(?:knowledge|consent|permission)"
        r"|account\s+(?:hacked|compromised|breached)|identity\s+theft"
        r"|someone\s+(?:else\s+)?(?:used|accessed|placed|hacked)"
        r"|(?:card|account)\s+(?:was\s+)?stolen)\b",
        re.IGNORECASE)),
    ("safety_or_injury", re.compile(
        r"\b(injur(?:y|ed|ies)|wounded|bleeding|hospitali[sz]ed"
        r"|electric\s+shock|electrocut(?:ed|ion)?|caught\s+fire"
        r"|burst\s+into\s+flames|explod(?:e|ed|ing)|burn(?:ed|t|ing)"
        r"|hazard(?:ous)?|dangerous|unsafe)\b",
        re.IGNORECASE)),
    ("legal_or_regulatory", re.compile(
        r"\b(lawsuit|sue\s+(?:you|the\s+company|them)|legal\s+action"
        r"|my\s+(?:lawyer|attorney)|consumer\s+(?:court|forum|protection)"
        r"|file\s+(?:an?\s+)?(?:fir|police\s+complaint|complaint\s+with\s+(?:the\s+)?police)"
        r"|report(?:ing)?\s+to\s+(?:the\s+)?(?:authorities|police)|ombudsman)\b",
        re.IGNORECASE)),
]


def detect_safety_escalation(message: str) -> str | None:
    """
    Return a category label if the message contains a high-severity safety,
    fraud, or legal signal that must be escalated to a specialist and never
    resolved autonomously. Returns None otherwise. Pure function, no side effects.
    """
    if not message:
        return None
    for label, pattern in _SAFETY_ESCALATION_PATTERNS:
        if pattern.search(message):
            return label
    return None


# ---------------------------------------------------------------------------
# Sensitive-data redaction (compliance / PII hygiene)
# ---------------------------------------------------------------------------
# AtlasCare never needs a customer's card number, CVV, email, or phone to service
# an order, so masking them in the customer's message before it reaches the LLM,
# the conversation checkpointer, or the trace/audit logs is a safe PCI/PII control.
# Deterministic — no model in the loop, so it cannot be prompted around.

# Card: 13–19 digits, optionally separated by spaces or dashes. Luhn-validated below
# so order totals / pincodes / order IDs are never mistaken for a card number.
_CARD_RE  = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
# CVV: only when a context word makes it unambiguous (avoids masking any 3–4 digits).
_CVV_RE   = re.compile(r"\b(?:cvv|cvc|cvv2|security\s*code)\b\s*[:#-]?\s*\d{3,4}\b",
                       re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Indian mobile: optional +91, then a 10-digit number starting 6–9. The negative
# look-arounds stop it matching inside a longer digit run.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)")


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum — true for valid payment-card numbers. Keeps the card regex
    from masking arbitrary long digit strings that aren't cards."""
    total, alt = 0, False
    for ch in reversed(digits):
        if not ch.isdigit():
            return False
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def redact_sensitive(text: str) -> tuple[str, list[str]]:
    """Mask card numbers, CVVs, emails, and phone numbers in `text`.

    Returns (redacted_text, kinds_found) where kinds_found is the de-duplicated,
    sorted list of categories that were masked (e.g. ['card', 'email']). Pure and
    idempotent — re-running on already-redacted text is a no-op.
    """
    if not text:
        return text, []

    found: set[str] = set()

    def _cvv(_m):
        found.add("cvv")
        return "[REDACTED_CVV]"

    def _card(m):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            found.add("card")
            return "[REDACTED_CARD]"
        return m.group()  # not a real card — leave untouched

    def _email(_m):
        found.add("email")
        return "[REDACTED_EMAIL]"

    def _phone(_m):
        found.add("phone")
        return "[REDACTED_PHONE]"

    # CVV (context-anchored) and cards first, before digit runs are tokenised away.
    text = _CVV_RE.sub(_cvv, text)
    text = _CARD_RE.sub(_card, text)
    text = _EMAIL_RE.sub(_email, text)
    text = _PHONE_RE.sub(_phone, text)
    return text, sorted(found)


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
        rules = [self._rule_no_over_limit_refund]
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
        # Strict (currency-marked) amounts plus bare amounts anchored to a refund
        # verb, so "refund 50000" (no ₹/Rs) is caught and fast-escalated pre-LLM.
        amounts = _extract_amounts(message) + _extract_refund_context_amounts(message)
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
    def _rule_no_over_limit_refund(
        self,
        execution_summary: list[dict],
    ) -> GuardrailVerdict:
        """
        RULE: GR-004 (CRITICAL)
        Final net for the auto-refund threshold: block the response if an
        autonomous refund was actually DISBURSED for an amount exceeding
        AUTO_REFUND_LIMIT_INR.

        "Disbursed" = a successful process_refund / cancel_item result that
        carries a 'refund' record with an amount. The threshold-escalation paths
        of either tool return a case_id and NO 'refund' record, so they are
        correctly treated as "no payment".

        WHY NOT "refund + escalation co-occurred": this rule used to block whenever
        ANY refund and ANY escalation happened in the same turn. That is a false
        positive for multi-order requests — e.g. a legitimate ₹18,000 refund on one
        order while an *unrelated* ₹42,000 order is correctly escalated (trace
        trc-bb9d77683993). The genuine policy violation is solely an over-limit
        disbursement, so we check the disbursed amount directly.

        Defence-in-depth: PaymentTool._enforce_threshold blocks over-limit refunds
        at source and pre-guardrail GR-001 fires before the LLM. This is the final net.
        """
        for e in execution_summary:
            if not e.get("success") or e.get("tool") not in {"process_refund", "cancel_item"}:
                continue
            refund = (e.get("data") or {}).get("refund")
            if not isinstance(refund, dict):
                continue
            try:
                amount = float(refund.get("amount_inr", 0) or 0)
            except (TypeError, ValueError):
                continue
            if amount > AUTO_REFUND_LIMIT_INR:
                return GuardrailVerdict.block(
                    rule_id="GR-004",
                    reason=(
                        f"CRITICAL: autonomous refund of ₹{amount:,.2f} exceeds the "
                        f"₹{AUTO_REFUND_LIMIT_INR:,.0f} auto-refund limit."
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
    Extract all monetary amounts from free text, including worded forms.
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
    # F-08: also catch worded amounts like "twenty-five thousand rupees"
    amounts.extend(_parse_worded_amount(text))
    return amounts


def _extract_refund_context_amounts(text: str) -> list[float]:
    """Bare numeric amounts tied to a refund verb (no currency marker required),
    e.g. 'refund 50000'. Anchored to the verb so an order id ('ORD-10014') or a
    pincode elsewhere in the message is never mistaken for a refund amount."""
    out: list[float] = []
    for m in _REFUND_CONTEXT_AMOUNT_RE.finditer(text or ""):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out