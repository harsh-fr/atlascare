import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from utils.money import round_inr
from utils.payment_methods import REFUND_METHODS as _SUPPORTED_METHODS

logger = logging.getLogger(__name__)


class RefundValidationError(Exception):
    pass


class RefundService:
    def create_refund_record(
        self,
        order_id: str,
        amount_inr: float,
        method: str,
        customer_id: str,
        sla_days: int,
    ) -> dict[str, Any]:
        self.validate_refund(amount_inr=amount_inr, method=method)
        refund_id         = self._generate_refund_id(order_id)
        amount_normalised = round_inr(amount_inr)
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
            refund_id, order_id, amount_normalised, method, sla_days,
        )
        return record

    def validate_refund(self, amount_inr: float, method: str) -> None:
        if amount_inr is None or Decimal(str(amount_inr)) <= 0:
            raise RefundValidationError(
                f"Refund amount must be greater than zero. Got: {amount_inr}"
            )
        amount_dec = Decimal(str(amount_inr))
        if amount_dec != amount_dec.quantize(Decimal("0.01")):
            raise RefundValidationError(
                f"Refund amount has too many decimal places: {amount_inr}. "
                "Maximum 2 decimal places allowed."
            )
        if method not in _SUPPORTED_METHODS:
            raise RefundValidationError(
                f"Refund method '{method}' is not supported. "
                f"Supported methods: {sorted(_SUPPORTED_METHODS)}"
            )

    @staticmethod
    def _generate_refund_id(order_id: str) -> str:
        order_suffix = order_id.split("-")[-1] if "-" in order_id else order_id
        unique_part  = uuid.uuid4().hex[:8].upper()
        return f"REF-{order_suffix}-{unique_part}"
