"""
tools/payment_tool.py
=====================
Payments Gateway integration tool.

Responsibility
--------------
  Exposes a single typed async method:
    - process_refund() : initiate a refund via the payments gateway

Critical policy enforced HERE (defence-in-depth layer 2)
---------------------------------------------------------
  Refunds above Rs.25,000 MUST NEVER be processed autonomously.
  This check lives in THREE places:
    1. Guardrails.pre_check()   — blocks before LLM (layer 1)
    2. THIS tool               — blocks at call time (layer 2)
    3. Guardrails.post_check() — verifies after execution (layer 3)

  Even if layers 1 and 3 somehow fail, this tool will refuse.
  The threshold is read from env var AUTO_REFUND_LIMIT_INR.

Design principles
-----------------
- The threshold check is deterministic code — never an LLM decision.
- Gateway failures (timeouts, errors) are retried with exponential
  backoff up to MAX_RETRIES before raising PaymentGatewayError.
- All refund attempts are logged with amount, method, and outcome
  for compliance auditability.
- Returns plain dict only — no internal objects leak out.
"""

import asyncio
import logging
import os
import random
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from repositories.payment_repository import PaymentRepository
from services.refund_service import RefundService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Policy constants
# ---------------------------------------------------------------------------
AUTO_REFUND_LIMIT_INR: float = float(
    os.getenv("AUTO_REFUND_LIMIT_INR", "25000.0")
)

MAX_RETRIES: int = int(os.getenv("PAYMENT_MAX_RETRIES", "3"))
RETRY_BASE_DELAY_S: float = float(os.getenv("PAYMENT_RETRY_BASE_DELAY_S", "0.5"))


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PaymentError(Exception):
    """Base error for all payment tool failures."""

class RefundThresholdError(PaymentError):
    """
    Raised when a refund amount exceeds the autonomous processing limit.
    This is a POLICY violation — must never be swallowed silently.
    """

class PaymentGatewayError(PaymentError):
    """Gateway returned an error or timed out after all retries."""

class InvalidRefundMethodError(PaymentError):
    """Requested refund method is not supported."""

class InvalidRefundAmountError(PaymentError):
    """Refund amount is zero, negative, or otherwise invalid."""


# ---------------------------------------------------------------------------
# PaymentTool
# ---------------------------------------------------------------------------
class PaymentTool:
    """
    Typed async interface to the Payments Gateway.

    Backed today by JSON config + simulated gateway behaviour;
    stable interface supports migration to a live gateway API.
    """

    _SUPPORTED_METHODS = {
        "HDFC_CREDIT",
        "ICICI_DEBIT",
        "SBI_NETBANKING",
        "UPI",
        "original",
    }

    def __init__(self) -> None:
        self._payment_repo = PaymentRepository()
        self._refund_svc   = RefundService()
        self._config       = self._payment_repo.get_config()
        logger.debug("PaymentTool initialised | config=%s", self._config)

    # ------------------------------------------------------------------
    # process_refund
    # ------------------------------------------------------------------
    async def process_refund(
        self,
        order_id: str,
        amount_inr: float,
        method: str,
        customer_id: str,
    ) -> dict[str, Any]:
        """
        Initiate a refund via the payments gateway.

        Parameters
        ----------
        order_id    : order being refunded
        amount_inr  : refund amount in INR (must be > 0)
        method      : payment method enum string
        customer_id : for audit logging

        Returns
        -------
        dict with keys: refund_id, order_id, amount_inr, method,
                        status, sla_days, message

        Raises
        ------
        RefundThresholdError      if amount > AUTO_REFUND_LIMIT_INR
        InvalidRefundAmountError  if amount <= 0
        InvalidRefundMethodError  if method not in supported set
        PaymentGatewayError       if gateway fails after all retries
        """
        # ----------------------------------------------------------
        # Step 1 — Input validation (deterministic, pre-gateway)
        # ----------------------------------------------------------
        self._validate_amount(amount_inr)
        self._validate_method(method)

        # ----------------------------------------------------------
        # Step 2 — THRESHOLD ENFORCEMENT (CRITICAL — layer 2)
        #
        # This check MUST happen in code before any gateway call.
        # It cannot be bypassed by the LLM, prompt, or planner.
        # ----------------------------------------------------------
        self._enforce_threshold(amount_inr, order_id, customer_id)

        # ----------------------------------------------------------
        # Step 3 — Gateway call with retry
        # ----------------------------------------------------------
        logger.info(
            "PaymentTool.process_refund | order=%s | amount=%.2f | "
            "method=%s | customer=%s",
            order_id,
            amount_inr,
            method,
            customer_id,
        )

        result = await self._call_gateway_with_retry(
            order_id=order_id,
            amount_inr=amount_inr,
            method=method,
            customer_id=customer_id,
        )

        logger.info(
            "PaymentTool.process_refund SUCCESS | refund_id=%s | "
            "order=%s | amount=%.2f | sla_days=%d",
            result["refund_id"],
            order_id,
            amount_inr,
            result["sla_days"],
        )

        return result

    # ------------------------------------------------------------------
    # Private — validation
    # ------------------------------------------------------------------
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

    def _enforce_threshold(
        self,
        amount_inr: float,
        order_id: str,
        customer_id: str,
    ) -> None:
        """
        CRITICAL POLICY CHECK — layer 2 of 3.

        Raises RefundThresholdError unconditionally if amount exceeds
        the autonomous refund limit. This method must NEVER be removed
        or weakened without a compliance review.
        """
        limit = self._config.get("auto_refund_limit_inr", AUTO_REFUND_LIMIT_INR)
        # Use Decimal for exact comparison — no float precision risk
        amount_dec = Decimal(str(amount_inr)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        limit_dec = Decimal(str(limit)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        if amount_dec > limit_dec:
            logger.warning(
                "THRESHOLD EXCEEDED | order=%s | customer=%s | "
                "amount=%.2f | limit=%.2f — BLOCKING autonomous refund",
                order_id,
                customer_id,
                amount_inr,
                limit,
            )
            raise RefundThresholdError(
                f"Refund of ₹{amount_inr:,.2f} exceeds the autonomous "
                f"processing limit of ₹{limit:,.0f}. "
                "This request must be escalated to a human specialist."
            )

    # ------------------------------------------------------------------
    # Private — gateway call with retry
    # ------------------------------------------------------------------
    async def _call_gateway_with_retry(
        self,
        order_id: str,
        amount_inr: float,
        method: str,
        customer_id: str,
    ) -> dict[str, Any]:
        """
        Call the simulated payments gateway with exponential backoff retry.

        Gateway failure behaviour is driven by payment_config.json
        (failure_rate, failure_code, failure_message) to simulate the
        realistic 3% timeout rate described in the config schema.
        """
        failure_rate = self._config.get("behaviour", {}).get("failure_rate", 0.03)
        failure_code = self._config.get("behaviour", {}).get("failure_code", "504")
        sla_days     = self._config.get("refund_sla_days", 5)

        last_error: Exception | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            t0 = time.monotonic()

            # Simulate gateway timeout based on configured failure rate
            if random.random() < failure_rate:
                latency_ms = int((time.monotonic() - t0) * 1000)
                logger.warning(
                    "Gateway simulated timeout | order=%s | attempt=%d/%d | "
                    "code=%s | latency_ms=%d",
                    order_id,
                    attempt,
                    MAX_RETRIES,
                    failure_code,
                    latency_ms,
                )
                last_error = PaymentGatewayError(
                    f"Gateway timeout (simulated {failure_code}) on attempt {attempt}."
                )
                if attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                    logger.info(
                        "Retrying payment in %.2fs | order=%s | attempt=%d",
                        delay,
                        order_id,
                        attempt,
                    )
                    await asyncio.sleep(delay)
                continue

            # Gateway success — build and persist refund record
            refund = self._refund_svc.create_refund_record(
                order_id=order_id,
                amount_inr=amount_inr,
                method=method,
                customer_id=customer_id,
                sla_days=sla_days,
            )
            self._payment_repo.save_refund(refund)

            return {
                "refund_id": refund["refund_id"],
                "order_id":  order_id,
                "amount_inr": amount_inr,
                "method":    method,
                "status":    "initiated",
                "sla_days":  sla_days,
                "message": (
                    f"Your refund of ₹{amount_inr:,.2f} has been initiated "
                    f"and will reflect in your account within {sla_days} "
                    f"business days."
                ),
            }

        # All retries exhausted
        raise PaymentGatewayError(
            f"Payment gateway failed after {MAX_RETRIES} attempts "
            f"for order '{order_id}'. Last error: {last_error}"
        )