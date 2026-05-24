"""
services/escalation_service.py
================================
Escalation business logic layer.

Responsibility
--------------
  Owns all business rules for CRM case creation:
    - create_case()           : build and persist a structured escalation case
    - build_handoff_summary() : generate a structured handoff description
                                for the human specialist agent

Design principles
-----------------
- Case ID generation is deterministic and collision-resistant.
- The handoff summary is built in code — never by the LLM —
  so it is consistent, auditable, and machine-parseable.
- trace_id linkage is MANDATORY — cases without a trace_id are
  rejected to ensure every case is linked to its agent interaction.
- All timestamps are UTC ISO 8601.
- Returns plain dicts matching the crm_cases.json schema.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from repositories.crm_repository import CrmRepository

logger = logging.getLogger(__name__)

# Valid priority levels matching crm_cases.json schema
_VALID_PRIORITIES = {"low", "medium", "high"}

# Valid case statuses
_VALID_STATUSES = {"open", "in_progress", "resolved", "closed"}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class EscalationError(Exception):
    """Raised when a case cannot be created due to a business rule violation."""


# ---------------------------------------------------------------------------
# EscalationService
# ---------------------------------------------------------------------------
class EscalationService:
    """
    Business logic for escalation case creation.

    Unlike OrderService and RefundService, EscalationService owns
    persistence directly (via CrmRepository) because case creation
    is its sole mutation concern and the CrmTool delegates entirely
    to this service.
    """

    def __init__(self) -> None:
        self._crm_repo = CrmRepository()
        logger.debug("EscalationService initialised.")

    # ------------------------------------------------------------------
    # create_case
    # ------------------------------------------------------------------
    def create_case(
        self,
        customer_id: str,
        order_id: str,
        reason: str,
        amount_inr: float | None,
        trace_id: str,
        priority: str = "medium",
    ) -> dict[str, Any]:
        """
        Create a new CRM escalation case with a structured handoff summary.

        Parameters
        ----------
        customer_id : authenticated customer
        order_id    : order the case relates to
        reason      : human-readable reason for escalation
        amount_inr  : refund amount in INR if applicable (None otherwise)
        trace_id    : MANDATORY — agent trace_id for audit linkage
        priority    : "low" | "medium" | "high" (default: "medium")

        Returns
        -------
        Complete case dict matching crm_cases.json schema.

        Raises
        ------
        EscalationError  if trace_id is missing or priority is invalid.
        """
        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------
        if not trace_id or not trace_id.strip():
            raise EscalationError(
                "trace_id is required for case creation. "
                "Every escalation case must be linked to an agent trace."
            )

        if priority not in _VALID_PRIORITIES:
            raise EscalationError(
                f"Invalid priority '{priority}'. "
                f"Must be one of: {sorted(_VALID_PRIORITIES)}"
            )

        if not customer_id or not order_id or not reason:
            raise EscalationError(
                "customer_id, order_id, and reason are all required."
            )

        # ----------------------------------------------------------
        # Build case
        # ----------------------------------------------------------
        case_id     = self._generate_case_id()
        created_at  = datetime.now(timezone.utc).isoformat()
        description = self.build_handoff_summary(
            customer_id=customer_id,
            order_id=order_id,
            reason=reason,
            amount_inr=amount_inr,
            trace_id=trace_id,
            priority=priority,
        )

        case = {
            "case_id":     case_id,
            "customer_id": customer_id,
            "order_id":    order_id,
            "status":      "open",
            "priority":    priority,
            "description": description,
            "amount_inr":  amount_inr,
            "trace_id":    trace_id,
            "created_at":  created_at,
        }

        # ----------------------------------------------------------
        # Persist
        # ----------------------------------------------------------
        self._crm_repo.save_case(case)

        logger.info(
            "EscalationService.create_case | case_id=%s | customer=%s | "
            "order=%s | priority=%s | amount=%s | trace=%s",
            case_id,
            customer_id,
            order_id,
            priority,
            f"₹{amount_inr:,.2f}" if amount_inr is not None else "N/A",
            trace_id,
        )

        return case

    # ------------------------------------------------------------------
    # build_handoff_summary
    # ------------------------------------------------------------------
    def build_handoff_summary(
        self,
        customer_id: str,
        order_id: str,
        reason: str,
        amount_inr: float | None,
        trace_id: str,
        priority: str,
    ) -> str:
        """
        Build a structured handoff description for the human specialist.

        This is intentionally deterministic — NOT generated by the LLM —
        so that the format is consistent, machine-parseable, and auditable.

        Format
        ------
        [ESCALATION CASE]
        Customer  : CUST-001
        Order     : ORD-78321
        Priority  : high
        Amount    : ₹42,000.00
        Reason    : Customer requesting full refund for damaged laptop.
                    Exceeds auto-refund threshold (₹25,000).
        Trace ID  : trc-a1b2c3d4
        Action    : Requires specialist review and manual refund approval.
        """
        amount_str = (
            f"₹{amount_inr:,.2f}" if amount_inr is not None else "N/A"
        )

        # Determine action guidance based on amount vs threshold
        from agent.guardrails import AUTO_REFUND_LIMIT_INR
        if amount_inr is not None and amount_inr > AUTO_REFUND_LIMIT_INR:
            action = (
                f"Refund of {amount_str} exceeds the ₹{AUTO_REFUND_LIMIT_INR:,.0f} "
                "auto-refund threshold. Requires specialist review and manual "
                "refund approval via the Payments portal."
            )
        else:
            action = (
                "Requires specialist review. Please contact the customer "
                "to resolve their query."
            )

        lines = [
            "[ESCALATION CASE — AtlasCare]",
            f"Customer  : {customer_id}",
            f"Order     : {order_id}",
            f"Priority  : {priority.upper()}",
            f"Amount    : {amount_str}",
            f"Reason    : {reason}",
            f"Action    : {action}",
            f"Trace ID  : {trace_id}",
            f"Created   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_case_id() -> str:
        """
        Generate a collision-resistant case ID.

        Format: CASE-<6 uppercase alphanumeric chars>
        Example: CASE-A1B2C3

        Matches the pattern ^CASE-[A-Z0-9]{6}$ in crm_cases.json schema.
        """
        suffix = uuid.uuid4().hex[:6].upper()
        return f"CASE-{suffix}"