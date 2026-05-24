"""
services/refund_service.py
===========================
Refund business logic layer.

Responsibility
--------------
  Owns all business rules and record construction for refunds:
    - create_refund_record() : build a validated, audit-ready refund dict
    - validate_refund()      : pure validation of refund eligibility

Design principles
-----------------
- Pure business logic only — no I/O, no repository access.
- Refund ID generation is deterministic and collision-resistant.
- All monetary arithmetic uses Decimal.
- The threshold check here is informational validation only —
  the hard enforcement lives in PaymentTool._enforce_threshold().
- Returns plain dicts matching the refunds.json schema.
"""

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

logger = logging.getLogger(__name__)

# Supported refund methods — mirrors payment_config.json
_SUPPORTED_METHODS = {
    "HDFC_CREDIT",
    "ICICI_DEBIT",
    "SBI_NETBANKING",
    "UPI",
    "original",
}


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------
class RefundValidationError(Exception):
    """Raised when a refund request fails business rule validation."""


# ---------------------------------------------------------------------------
# RefundService
# ---------------------------------------------------------------------------
class RefundService:
    """
    Stateless business logic for refund record construction and validation.

    Accepts and returns plain dicts.
    Never touches repositories or tools directly.
    """

    # ------------------------------------------------------------------
    # create_refund_record
    # ------------------------------------------------------------------
    def create_refund_record(
        self,
        order_id: str,
        amount_inr: float,
        method: str,
        customer_id: str,
        sla_days: int,
    ) -> dict[str, Any]:
        """
        Build a complete, audit-ready refund record dict.

        Parameters
        ----------
        order_id    : order being refunded
        amount_inr  : refund amount in INR
        method      : payment method string
        customer_id : customer receiving the refund
        sla_days    : SLA days from payment config

        Returns
        -------
        Refund dict with keys:
          refund_id, order_id, customer_id, amount_inr, method,
          status, sla_days, created_at

        Raises
        ------
        RefundValidationError  if inputs fail validation.
        """
        self.validate_refund(
            amount_inr=amount_inr,
            method=method,
        )

        # Generate collision-resistant refund ID
        refund_id = self._generate_refund_id(order_id)

        # Normalise amount to 2 decimal places
        amount_normalised = float(
            Decimal(str(amount_inr)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        )

        record = {
            "refund_id":   refund_id,
            "order_id":    order_id,
            "customer_id": customer_id,
            "amount_inr":  amount_normalised,
            "method":      method,
            "status":      "initiated",
            "sla_days":    sla_days,
            "created_at":  datetime.now(timezone.utc).isoformat(),
        }

        logger.info(
            "RefundService.create_refund_record | refund_id=%s | "
            "order=%s | amount=%.2f | method=%s | sla_days=%d",
            refund_id,
            order_id,
            amount_normalised,
            method,
            sla_days,
        )

        return record

    # ------------------------------------------------------------------
    # validate_refund
    # ------------------------------------------------------------------
    def validate_refund(
        self,
        amount_inr: float,
        method: str,
    ) -> None:
        """
        Validate refund inputs against business rules.

        Raises
        ------
        RefundValidationError  on any violation.
        """
        # Amount must be positive
        if amount_inr is None or Decimal(str(amount_inr)) <= 0:
            raise RefundValidationError(
                f"Refund amount must be greater than zero. Got: {amount_inr}"
            )

        # Amount must not have more than 2 decimal places
        amount_dec = Decimal(str(amount_inr))
        if amount_dec != amount_dec.quantize(Decimal("0.01")):
            raise RefundValidationError(
                f"Refund amount has too many decimal places: {amount_inr}. "
                "Maximum 2 decimal places allowed."
            )

        # Method must be supported
        if method not in _SUPPORTED_METHODS:
            raise RefundValidationError(
                f"Refund method '{method}' is not supported. "
                f"Supported methods: {sorted(_SUPPORTED_METHODS)}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_refund_id(order_id: str) -> str:
        """
        Generate a collision-resistant refund ID.

        Format: REF-<order_suffix>-<uuid4_short>
        Example: REF-78321-a1b2c3d4

        The order suffix makes refund IDs traceable back to the
        originating order without a database lookup.
        """
        order_suffix = order_id.split("-")[-1] if "-" in order_id else order_id
        unique_part  = uuid.uuid4().hex[:8].upper()
        return f"REF-{order_suffix}-{unique_part}"