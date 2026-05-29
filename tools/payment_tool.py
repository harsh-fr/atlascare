import asyncio
import logging
import os
import random
import time
from typing import Any

from utils.money import to_decimal
from utils.payment_methods import REFUND_METHODS, DEFAULT_AUTO_REFUND_LIMIT_INR
from repositories.payment_repository import PaymentRepository
from services.refund_service import RefundService

logger = logging.getLogger(__name__)

AUTO_REFUND_LIMIT_INR: float = float(
    os.getenv("AUTO_REFUND_LIMIT_INR", str(DEFAULT_AUTO_REFUND_LIMIT_INR))
)
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
    # Code-level ceiling of refund destinations the gateway can render/route.
    # The LIVE supported set is narrowed to this ∩ payment_config.supported_methods
    # in __init__ — see _resolve_supported_methods. Kept as the class default so a
    # PaymentTool built without config still validates safely.
    _SUPPORTED_METHODS = REFUND_METHODS

    def __init__(self) -> None:
        self._payment_repo = PaymentRepository()
        self._refund_svc   = RefundService()
        self._config       = self._payment_repo.get_config()
        self._SUPPORTED_METHODS = self._resolve_supported_methods(self._config)
        logger.debug(
            "PaymentTool initialised | supported_methods=%s | config=%s",
            sorted(self._SUPPORTED_METHODS), self._config,
        )

    @staticmethod
    def _resolve_supported_methods(config: dict[str, Any]) -> frozenset[str]:
        """Live refund-destination set = code ceiling ∩ payment_config.supported_methods.

        payment_config.json is the source of truth for which rails are enabled, but
        code is the ceiling: a method must already be one REFUND_METHODS can label
        and route, so config can only ever *narrow* the set, never introduce a rail
        the code can't honour. 'original' (refund to source) is always permitted —
        it is the safe default, not a gateway rail. If config omits supported_methods
        entirely we fall back to the full code set rather than disabling all refunds.
        """
        configured = config.get("supported_methods")
        if not configured:
            return REFUND_METHODS
        enabled = REFUND_METHODS & set(configured)
        unknown = set(configured) - REFUND_METHODS - {"original"}
        if unknown:
            logger.warning(
                "payment_config.supported_methods lists methods the code cannot "
                "route (ignored): %s", sorted(unknown),
            )
        return frozenset(enabled | {"original"})

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

        # _call_gateway_with_retry owns the single retry/backoff loop and raises
        # PaymentGatewayError once all attempts are exhausted. (It used to be
        # wrapped in a second retry loop here, which silently multiplied the
        # attempt count to MAX_RETRIES**2 — removed.)
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
