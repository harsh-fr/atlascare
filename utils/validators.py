import re
from typing import Any

from utils.payment_methods import PAYMENT_METHODS_WITH_COD

_ORDER_ID_RE    = re.compile(r"^ORD-\d{5}$")
_CUSTOMER_ID_RE = re.compile(r"^CUST-\d{3}$")
_CASE_ID_RE     = re.compile(r"^CASE-[A-Z0-9]{6}$")
_KB_ARTICLE_RE  = re.compile(r"^KB-\d{3}$")

_VALID_ORDER_STATUSES  = {"placed", "processing", "shipped", "delivered", "cancelled"}
_VALID_ITEM_STATUSES   = {"active", "cancelled"}
_VALID_CASE_STATUSES   = {"open", "in_progress", "resolved", "closed"}
_VALID_PRIORITIES      = {"low", "medium", "high"}
_VALID_CUSTOMER_TIERS  = {"standard", "silver", "gold", "platinum"}
_VALID_PAYMENT_METHODS = set(PAYMENT_METHODS_WITH_COD)

_MAX_REFUND_AMOUNT = 10_000_000.0
_MIN_REFUND_AMOUNT = 0.01


def _id_validator(field_name: str, pattern: re.Pattern, hint: str):
    def validate(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string, got {type(value).__name__}.")
        normalised = value.strip().upper()
        if not pattern.match(normalised):
            raise ValueError(f"Invalid {field_name} '{value}'. Expected format: {hint}.")
        return normalised
    validate.__name__ = f"validate_{field_name}"
    return validate


def _enum_validator(display_name: str, valid_set: set):
    def validate(value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{display_name} must be a string, got {type(value).__name__}.")
        if value not in valid_set:
            raise ValueError(
                f"Invalid {display_name} '{value}'. Must be one of: {sorted(valid_set)}."
            )
        return value
    validate.__name__ = f"validate_{display_name.replace(' ', '_')}"
    return validate


validate_order_id      = _id_validator("order_id",    _ORDER_ID_RE,    "ORD-XXXXX (e.g. ORD-99999)")
validate_customer_id   = _id_validator("customer_id", _CUSTOMER_ID_RE, "CUST-XXX (e.g. CUST-001)")
validate_case_id       = _id_validator("case_id",     _CASE_ID_RE,     "CASE-XXXXXX (e.g. CASE-A1B2C3)")
validate_kb_article_id = _id_validator("article_id",  _KB_ARTICLE_RE,  "KB-XXX (e.g. KB-001)")

validate_payment_method = _enum_validator("payment_method", _VALID_PAYMENT_METHODS)
validate_priority       = _enum_validator("priority",       _VALID_PRIORITIES)
validate_order_status   = _enum_validator("order_status",   _VALID_ORDER_STATUSES)
validate_customer_tier  = _enum_validator("customer_tier",  _VALID_CUSTOMER_TIERS)


def validate_refund_amount(amount: Any) -> float:
    if not isinstance(amount, (int, float)):
        raise ValueError(f"Refund amount must be numeric, got {type(amount).__name__}.")
    amount_f = float(amount)
    if amount_f < _MIN_REFUND_AMOUNT:
        raise ValueError(f"Refund amount ₹{amount_f} is below minimum ₹{_MIN_REFUND_AMOUNT}.")
    if amount_f > _MAX_REFUND_AMOUNT:
        raise ValueError(
            f"Refund amount ₹{amount_f:,.2f} exceeds sanity cap ₹{_MAX_REFUND_AMOUNT:,.0f}."
        )
    return round(amount_f, 2)


def validate_address(address: Any) -> dict:
    if not isinstance(address, dict):
        raise ValueError(f"address must be a dict, got {type(address).__name__}.")
    required = ["line1", "city", "state", "pincode"]
    missing  = [k for k in required if not address.get(k, "").strip()]
    if missing:
        raise ValueError(f"Address is missing required fields: {missing}.")
    return address


def validate_line_id(line_id: Any) -> int:
    try:
        line_id_int = int(line_id)
    except (TypeError, ValueError):
        raise ValueError(f"line_id must be an integer, got '{line_id}'.")
    if line_id_int < 1:
        raise ValueError(f"line_id must be >= 1 (1-indexed), got {line_id_int}.")
    return line_id_int
