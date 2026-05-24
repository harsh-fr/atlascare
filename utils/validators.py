"""
utils/validators.py
====================
Shared validation utilities.

Responsibility
--------------
  Pure functions for validating common data types used across
  the AtlasCare codebase:
    - Order IDs, customer IDs, case IDs
    - Refund amounts
    - Payment methods
    - Priority levels
    - Address completeness

Design principles
-----------------
- All functions are pure — no side effects, no I/O.
- Raise ValueError with a clear message on failure.
- Return the (possibly normalised) value on success.
- Centralised here so validation logic is never duplicated
  across tools, services, or models.
"""

import re
from typing import Any

# ---------------------------------------------------------------------------
# ID patterns — match crm_cases.json and orders.json schemas exactly
# ---------------------------------------------------------------------------
_ORDER_ID_RE    = re.compile(r"^ORD-\d{5}$")
_CUSTOMER_ID_RE = re.compile(r"^CUST-\d{3}$")
_CASE_ID_RE     = re.compile(r"^CASE-[A-Z0-9]{6}$")
_KB_ARTICLE_RE  = re.compile(r"^KB-\d{3}$")

# Valid enums
_VALID_ORDER_STATUSES   = {"placed", "processing", "shipped", "delivered", "cancelled"}
_VALID_ITEM_STATUSES    = {"active", "cancelled"}
_VALID_CASE_STATUSES    = {"open", "in_progress", "resolved", "closed"}
_VALID_PRIORITIES       = {"low", "medium", "high"}
_VALID_CUSTOMER_TIERS   = {"standard", "silver", "gold", "platinum"}
_VALID_PAYMENT_METHODS  = {
    "HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original", "COD"
}

# Monetary limits
_MAX_REFUND_AMOUNT = 10_000_000.0   # 1 crore — sanity upper bound
_MIN_REFUND_AMOUNT = 0.01


# ---------------------------------------------------------------------------
# ID validators
# ---------------------------------------------------------------------------
def validate_order_id(order_id: Any) -> str:
    """
    Validate and return a normalised order ID.

    Valid format: ORD-XXXXX  (5 digits)
    Example     : ORD-78321

    Raises ValueError on invalid input.
    """
    if not isinstance(order_id, str):
        raise ValueError(
            f"order_id must be a string, got {type(order_id).__name__}."
        )
    normalised = order_id.strip().upper()
    if not _ORDER_ID_RE.match(normalised):
        raise ValueError(
            f"Invalid order_id '{order_id}'. "
            "Expected format: ORD-XXXXX (e.g. ORD-78321)."
        )
    return normalised


def validate_customer_id(customer_id: Any) -> str:
    """
    Validate and return a normalised customer ID.

    Valid format: CUST-XXX  (3 digits)
    Example     : CUST-001

    Raises ValueError on invalid input.
    """
    if not isinstance(customer_id, str):
        raise ValueError(
            f"customer_id must be a string, got {type(customer_id).__name__}."
        )
    normalised = customer_id.strip().upper()
    if not _CUSTOMER_ID_RE.match(normalised):
        raise ValueError(
            f"Invalid customer_id '{customer_id}'. "
            "Expected format: CUST-XXX (e.g. CUST-001)."
        )
    return normalised


def validate_case_id(case_id: Any) -> str:
    """
    Validate and return a normalised CRM case ID.

    Valid format: CASE-XXXXXX  (6 uppercase alphanumeric)
    Example     : CASE-A1B2C3

    Raises ValueError on invalid input.
    """
    if not isinstance(case_id, str):
        raise ValueError(
            f"case_id must be a string, got {type(case_id).__name__}."
        )
    normalised = case_id.strip().upper()
    if not _CASE_ID_RE.match(normalised):
        raise ValueError(
            f"Invalid case_id '{case_id}'. "
            "Expected format: CASE-XXXXXX (e.g. CASE-A1B2C3)."
        )
    return normalised


def validate_kb_article_id(article_id: Any) -> str:
    """
    Validate and return a normalised KB article ID.

    Valid format: KB-XXX  (3 digits)
    Example     : KB-001

    Raises ValueError on invalid input.
    """
    if not isinstance(article_id, str):
        raise ValueError(
            f"article_id must be a string, got {type(article_id).__name__}."
        )
    normalised = article_id.strip().upper()
    if not _KB_ARTICLE_RE.match(normalised):
        raise ValueError(
            f"Invalid article_id '{article_id}'. "
            "Expected format: KB-XXX (e.g. KB-001)."
        )
    return normalised


# ---------------------------------------------------------------------------
# Monetary validators
# ---------------------------------------------------------------------------
def validate_refund_amount(amount: Any) -> float:
    """
    Validate a refund amount.

    Rules
    -----
    - Must be numeric (int or float)
    - Must be >= 0.01
    - Must be <= 10,000,000 (sanity cap)
    - Must have at most 2 decimal places

    Returns normalised float rounded to 2 decimal places.
    Raises ValueError on any violation.
    """
    if not isinstance(amount, (int, float)):
        raise ValueError(
            f"Refund amount must be numeric, got {type(amount).__name__}."
        )
    amount_f = float(amount)

    if amount_f < _MIN_REFUND_AMOUNT:
        raise ValueError(
            f"Refund amount ₹{amount_f} is below minimum ₹{_MIN_REFUND_AMOUNT}."
        )
    if amount_f > _MAX_REFUND_AMOUNT:
        raise ValueError(
            f"Refund amount ₹{amount_f:,.2f} exceeds sanity cap "
            f"₹{_MAX_REFUND_AMOUNT:,.0f}."
        )

    # Round to 2 decimal places
    rounded = round(amount_f, 2)
    return rounded


# ---------------------------------------------------------------------------
# Enum validators
# ---------------------------------------------------------------------------
def validate_payment_method(method: Any) -> str:
    """
    Validate a payment method string.
    Raises ValueError if not in the supported set.
    """
    if not isinstance(method, str):
        raise ValueError(
            f"payment_method must be a string, got {type(method).__name__}."
        )
    if method not in _VALID_PAYMENT_METHODS:
        raise ValueError(
            f"Unsupported payment method '{method}'. "
            f"Must be one of: {sorted(_VALID_PAYMENT_METHODS)}."
        )
    return method


def validate_priority(priority: Any) -> str:
    """
    Validate a case priority string.
    Raises ValueError if not in {low, medium, high}.
    """
    if not isinstance(priority, str):
        raise ValueError(
            f"priority must be a string, got {type(priority).__name__}."
        )
    if priority not in _VALID_PRIORITIES:
        raise ValueError(
            f"Invalid priority '{priority}'. "
            f"Must be one of: {sorted(_VALID_PRIORITIES)}."
        )
    return priority


def validate_order_status(status: Any) -> str:
    """Validate an order status string."""
    if status not in _VALID_ORDER_STATUSES:
        raise ValueError(
            f"Invalid order status '{status}'. "
            f"Must be one of: {sorted(_VALID_ORDER_STATUSES)}."
        )
    return status


def validate_customer_tier(tier: Any) -> str:
    """Validate a customer tier string."""
    if tier not in _VALID_CUSTOMER_TIERS:
        raise ValueError(
            f"Invalid customer tier '{tier}'. "
            f"Must be one of: {sorted(_VALID_CUSTOMER_TIERS)}."
        )
    return tier


# ---------------------------------------------------------------------------
# Address validator
# ---------------------------------------------------------------------------
def validate_address(address: Any) -> dict:
    """
    Validate that an address dict has all required fields populated.

    Required keys: line1, city, state, pincode
    Raises ValueError if any required field is missing or empty.
    """
    if not isinstance(address, dict):
        raise ValueError(
            f"address must be a dict, got {type(address).__name__}."
        )
    required = ["line1", "city", "state", "pincode"]
    missing = [k for k in required if not address.get(k, "").strip()]
    if missing:
        raise ValueError(
            f"Address is missing required fields: {missing}."
        )
    return address


# ---------------------------------------------------------------------------
# Line item validator
# ---------------------------------------------------------------------------
def validate_line_id(line_id: Any) -> int:
    """
    Validate a line item ID.
    Must be a positive integer (1-indexed per orders.json schema).
    """
    try:
        line_id_int = int(line_id)
    except (TypeError, ValueError):
        raise ValueError(
            f"line_id must be an integer, got '{line_id}'."
        )
    if line_id_int < 1:
        raise ValueError(
            f"line_id must be >= 1 (1-indexed), got {line_id_int}."
        )
    return line_id_int