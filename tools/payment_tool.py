import asyncio
import logging
import os
import random
import time
from typing import Any

from utils.money import to_decimal
from repositories.payment_repository import PaymentRepository
from services.refund_service import RefundService

logger = logging.getLogger(__name__)

AUTO_REFUND_LIMIT_INR: float = float(os.getenv("AUTO_REFUND_LIMIT_INR", "25000.0"))
MAX_RETRIES: int              = int(os.getenv("PAYMENT_MAX_RETRIES", "3"))
RETRY_BASE_DELAY_S: float     = float(os.getenv("PAYMENT_RETRY_BASE_DELAY_S", "0.5"))


class PaymentError(Exception):
    pass

class RefundThresholdError(PaymentError):
    """POLICY violation — must never be swallowed silently."""

class PaymentGatewayError(PaymentError):
    pass

class InvalidRefundMethodError(PaymentError):
    pass

class InvalidRefundAmountError(PaymentError):
    pass


class PaymentTool:
    _SUPPORTED_METHODS = {"HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"}

    def __init__(self) -> None:
        self._payment_repo = PaymentRepository()
        self._refund_svc   = RefundService()
        self._config       = self._payment_repo.get_config()
        logger.debug("PaymentTool initialised | config=%s", self._config)

    async def process_refund(
        self,
        order_id: str,
        amount_inr: float,
        method: str,
        customer_id: str,
    ) -> dict[str, Any]:
        self._validate_amount(amount_inr)
        self._validate_method(method)
        self._enforce_threshold(amount_inr, order_id, customer_id)

        logger.info(
            "PaymentTool.process_refund | order=%s | amount=%.2f | method=%s | customer=%s",
            order_id, amount_inr, method, customer_id,
        )

        import tools.payment_tool as _self_mod
        max_retries      = _self_mod.MAX_RETRIES
        retry_base_delay = _self_mod.RETRY_BASE_DELAY_S
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                result = await self._call_gateway_with_retry(
                    order_id=order_id,
                    amount_inr=amount_inr,
                    method=method,
                    customer_id=customer_id,
                )
                logger.info(
                    "PaymentTool.process_refund SUCCESS | refund_id=%s | "
                    "order=%s | amount=%.2f | sla_days=%d",
                    result["refund_id"], order_id, amount_inr, result["sla_days"],
                )
                return result
            except PaymentGatewayError as exc:
                last_error = exc
                logger.warning(
                    "process_refund attempt %d/%d failed | order=%s | error=%s",
                    attempt, max_retries, order_id, exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(retry_base_delay * (2 ** (attempt - 1)))

        raise PaymentGatewayError(
            f"Payment gateway failed after {max_retries} attempts "
            f"for order '{order_id}'. Last error: {last_error}"
        )

    def _validate_amount(self, amount_inr: float) -> None:
        if amount_inr is None or amount_inr <= 0:
            raise InvalidRefundAmountError(
                f"Refund amount must be greater than zero. Got: {amount_inr}"
            )

    def _validate_method(self, method: str) -> None:
        if method not in self._SUPPORTED_METHODS:
            raise InvalidRefundMethodError(
                f"Refund method '{method}' is not supported. "
                f"Supported: {sorted(self._SUPPORTED_METHODS)}"
            )

    def _enforce_threshold(self, amount_inr: float, order_id: str, customer_id: str) -> None:
        # Layer 2 of 3: never remove or weaken without a compliance review.
        limit = self._config.get("auto_refund_limit_inr", AUTO_REFUND_LIMIT_INR)
        if to_decimal(amount_inr) > to_decimal(limit):
            logger.warning(
                "THRESHOLD EXCEEDED | order=%s | customer=%s | "
                "amount=%.2f | limit=%.2f — BLOCKING autonomous refund",
                order_id, customer_id, amount_inr, limit,
            )
            raise RefundThresholdError(
                f"Refund of ₹{amount_inr:,.2f} for order '{order_id}' "
                "requires specialist review and cannot be processed automatically."
            )

    def get_total_refunded(self, order_id: str) -> float:
        refunds = self._payment_repo.find_refunds_by_order(order_id)
        return float(sum(r.get("amount_inr", 0.0) for r in refunds))

    async def _call_gateway_with_retry(
        self,
        order_id: str,
        amount_inr: float,
        method: str,
        customer_id: str,
    ) -> dict[str, Any]:
        failure_rate = self._config.get("behaviour", {}).get("failure_rate", 0.03)
        failure_code = self._config.get("behaviour", {}).get("failure_code", "504")
        sla_days     = self._config.get("refund_sla_days", 5)
        last_error: Exception | None = None

        # Read module-level constants at call time so patch.object() in tests can override them.
        import tools.payment_tool as _self_mod
        max_retries      = _self_mod.MAX_RETRIES
        retry_base_delay = _self_mod.RETRY_BASE_DELAY_S

        for attempt in range(1, max_retries + 1):
            t0 = time.monotonic()
            if random.random() < failure_rate:
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.warning(
                    "Gateway simulated timeout | order=%s | attempt=%d/%d | "
                    "code=%s | latency_ms=%d",
                    order_id, attempt, max_retries, failure_code, latency_ms,
                )
                last_error = PaymentGatewayError(
                    f"Gateway timeout (simulated {failure_code}) on attempt {attempt}."
                )
                if attempt < max_retries:
                    delay = retry_base_delay * (2 ** (attempt - 1))
                    logger.info(
                        "Retrying payment in %.2fs | order=%s | attempt=%d",
                        delay, order_id, attempt,
                    )
                    await asyncio.sleep(delay)
                continue

            refund = self._refund_svc.create_refund_record(
                order_id=order_id,
                amount_inr=amount_inr,
                method=method,
                customer_id=customer_id,
                sla_days=sla_days,
            )
            self._payment_repo.save_refund(refund)
            return {
                "refund_id":  refund["refund_id"],
                "order_id":   order_id,
                "amount_inr": amount_inr,
                "method":     method,
                "status":     "initiated",
                "sla_days":   sla_days,
                "message": (
                    f"Your refund of ₹{amount_inr:,.2f} has been initiated "
                    f"and will reflect in your account within {sla_days} "
                    f"business days."
                ),
            }

        raise PaymentGatewayError(
            f"Payment gateway failed after {max_retries} attempts "
            f"for order '{order_id}'. Last error: {last_error}"
        )
