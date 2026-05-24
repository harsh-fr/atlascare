"""
tools/crm_tool.py
=================
Customer Relationship Management (CRM) integration tool.

Responsibility
--------------
  Exposes typed, async methods the Executor calls to interact
  with customer and case data:
    - get_customer()    : fetch a customer profile by ID
    - create_case()     : create a new escalation case with audit trail
    - get_cases()       : retrieve existing cases for a customer

Design principles
-----------------
- Tools are the ONLY layer that touches repositories.
- Case creation is the critical path for J3 (escalation).
  It MUST record trace_id, structured handoff summary, and priority.
- Case IDs are generated deterministically (not by the LLM).
- All mutations go through the EscalationService for business logic.
- Returns plain dicts only — no repository objects leak out.
"""

import logging
from typing import Any

from repositories.crm_repository import CrmRepository
from services.escalation_service import EscalationService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class CrmError(Exception):
    """Base error for all CRM tool failures."""

class CustomerNotFoundError(CrmError):
    """Customer ID does not exist in the repository."""

class CaseCreationError(CrmError):
    """Case could not be created."""


# ---------------------------------------------------------------------------
# CrmTool
# ---------------------------------------------------------------------------
class CrmTool:
    """
    Typed async interface to the CRM system.

    Backed today by JSON repositories; stable interface supports
    migration to a live CRM API (Salesforce, Zendesk, etc.)
    without changing callers.
    """

    def __init__(self) -> None:
        self._crm_repo        = CrmRepository()
        self._escalation_svc  = EscalationService()
        logger.debug("CrmTool initialised.")

    # ------------------------------------------------------------------
    # get_customer
    # ------------------------------------------------------------------
    async def get_customer(self, customer_id: str) -> dict[str, Any]:
        """
        Fetch a customer profile by ID.

        Returns
        -------
        dict matching the customers schema in crm_cases.json.

        Raises
        ------
        CustomerNotFoundError  if customer_id does not exist.
        """
        logger.debug("CrmTool.get_customer | customer_id=%s", customer_id)

        customer = self._crm_repo.find_customer_by_id(customer_id)
        if customer is None:
            raise CustomerNotFoundError(
                f"Customer '{customer_id}' not found."
            )
        return customer

    # ------------------------------------------------------------------
    # create_case
    # ------------------------------------------------------------------
    async def create_case(
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

        This is the critical J3 path.

        Parameters
        ----------
        customer_id : authenticated customer
        order_id    : order the case relates to
        reason      : human-readable reason for escalation
        amount_inr  : refund amount in INR if applicable (None otherwise)
        trace_id    : agent trace_id — MUST be attached for audit linkage
        priority    : "low" | "medium" | "high"  (default: "medium")

        Returns
        -------
        dict with keys: case_id, customer_id, order_id, status,
                        priority, description, amount_inr, trace_id,
                        created_at

        Raises
        ------
        CustomerNotFoundError  if customer_id does not exist.
        CaseCreationError      if case could not be persisted.
        """
        logger.info(
            "CrmTool.create_case | customer=%s | order=%s | "
            "amount=%.2f | priority=%s | trace=%s",
            customer_id,
            order_id,
            amount_inr or 0.0,
            priority,
            trace_id,
        )

        # Validate customer exists before creating case
        customer = self._crm_repo.find_customer_by_id(customer_id)
        if customer is None:
            raise CustomerNotFoundError(
                f"Cannot create case: customer '{customer_id}' not found."
            )

        # Delegate case construction + persistence to EscalationService
        try:
            case = self._escalation_svc.create_case(
                customer_id=customer_id,
                order_id=order_id,
                reason=reason,
                amount_inr=amount_inr,
                trace_id=trace_id,
                priority=priority,
            )
        except Exception as exc:
            raise CaseCreationError(
                f"Failed to create CRM case for order '{order_id}': {exc}"
            ) from exc

        logger.info(
            "CrmTool.create_case SUCCESS | case_id=%s | customer=%s | "
            "order=%s | trace=%s",
            case["case_id"],
            customer_id,
            order_id,
            trace_id,
        )

        return case

    # ------------------------------------------------------------------
    # get_cases
    # ------------------------------------------------------------------
    async def get_cases(
        self,
        customer_id: str,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve existing CRM cases for a customer.

        Parameters
        ----------
        customer_id   : customer whose cases to retrieve
        status_filter : optional filter — "open" | "in_progress" |
                        "resolved" | "closed". None returns all.

        Returns
        -------
        List of case dicts, newest first.

        Raises
        ------
        CustomerNotFoundError  if customer_id does not exist.
        """
        logger.debug(
            "CrmTool.get_cases | customer_id=%s | filter=%s",
            customer_id,
            status_filter,
        )

        customer = self._crm_repo.find_customer_by_id(customer_id)
        if customer is None:
            raise CustomerNotFoundError(
                f"Customer '{customer_id}' not found."
            )

        cases = self._crm_repo.find_cases_by_customer(customer_id)

        if status_filter:
            cases = [c for c in cases if c.get("status") == status_filter]

        # Return newest first
        cases_sorted = sorted(
            cases,
            key=lambda c: c.get("created_at", ""),
            reverse=True,
        )

        logger.debug(
            "CrmTool.get_cases | customer=%s | found=%d",
            customer_id,
            len(cases_sorted),
        )

        return cases_sorted