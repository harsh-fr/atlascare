"""
utils/payment_methods.py
========================
Single source of truth for payment / refund method identifiers.

Previously the supported-method set, the human-readable labels, the
free-text alias map, and the auto-refund default were each duplicated
across tools/payment_tool.py, services/refund_service.py, utils/validators.py,
agent/graph.py and repositories/payment_repository.py — and could drift.
Everything method-related now lives here and is imported by those modules.
"""

import logging

logger = logging.getLogger(__name__)

# Electronic refund destinations the payment gateway accepts, plus the
# sentinel "original" (refund to the order's original payment instrument).
REFUND_METHODS: frozenset[str] = frozenset(
    {"HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"}
)

# Refund methods minus the "original" sentinel — the concrete electronic
# destinations a customer can name.
ELECTRONIC_REFUND_METHODS: frozenset[str] = REFUND_METHODS - {"original"}

# Every payment method that may appear on an order, including Cash on Delivery
# (which is a valid *order* payment method but never a valid *refund* target).
PAYMENT_METHODS_WITH_COD: frozenset[str] = REFUND_METHODS | {"COD"}

# Stable ordering for tool schema enums / error messages.
REFUND_METHOD_ENUM: list[str] = ["HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"]

# Human-readable names for internal codes. Used both to render replies and as a
# deterministic scrub backstop so an internal code can never leak to a customer.
INTERNAL_METHOD_LABELS: dict[str, str] = {
    "HDFC_CREDIT":    "HDFC Credit Card",
    "ICICI_DEBIT":    "ICICI Debit Card",
    "SBI_NETBANKING": "SBI Net Banking",
    "UPI":            "UPI",
}

# Free-text → canonical code. Covers what customers and the LLM actually type.
METHOD_ALIASES: dict[str, str] = {
    "hdfc_credit": "HDFC_CREDIT", "icici_debit": "ICICI_DEBIT",
    "sbi_netbanking": "SBI_NETBANKING", "upi": "UPI", "original": "original",
    "hdfc": "HDFC_CREDIT", "hdfc credit": "HDFC_CREDIT", "hdfc credit card": "HDFC_CREDIT",
    "hdfc card": "HDFC_CREDIT", "icici": "ICICI_DEBIT", "icici debit": "ICICI_DEBIT",
    "icici debit card": "ICICI_DEBIT", "icici card": "ICICI_DEBIT",
    "sbi": "SBI_NETBANKING", "sbi net banking": "SBI_NETBANKING",
    "net banking": "SBI_NETBANKING", "netbanking": "SBI_NETBANKING",
    "gpay": "UPI", "google pay": "UPI", "phonepe": "UPI", "paytm": "UPI",
    "original_payment": "original", "original_payment_method": "original",
    "original payment method": "original", "same card": "original",
    "same method": "original", "source": "original",
}

# Auto-refund threshold default (INR). The live value is resolved from
# payment_config.json first (see agent.guardrails); this is the safe fallback
# baked into code. NEVER change without a compliance review.
DEFAULT_AUTO_REFUND_LIMIT_INR: float = 25000.0


def normalise_refund_method(method: str) -> str:
    """Map free-text / code input to a canonical refund method.

    Unrecognised input falls back to 'original' (refund to source) and logs a
    warning unless the input was already an 'original' synonym or empty.
    """
    key = (method or "").lower().strip()
    mapped = METHOD_ALIASES.get(key)
    if mapped is None:
        if key and key != "original":
            logger.warning(
                "Unrecognized refund method %r — defaulting to 'original'.", method
            )
        return "original"
    return mapped
