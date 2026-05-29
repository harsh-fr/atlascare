import logging
from typing import Any

from repositories.crm_repository import CrmRepository
from services.escalation_service import EscalationService

logger = logging.getLogger(__name__)


class CrmError(Exception):
    pass

class CustomerNotFoundError(CrmError):
    pass

class CaseCreationError(CrmError):
    pass


class CrmTool:
    def __init__(self) -> None:
        self._crm_repo       = CrmRepository()
        self._escalation_svc = EscalationService()
        logger.debug("CrmTool initialised.")

    def _assert_customer_exists(self, customer_id: str) -> dict:
        customer = self._crm_repo.find_customer_by_id(customer_id)
        if customer is None:
            raise CustomerNotFoundError(f"Customer '{customer_id}' not found.")
        return customer

    async def get_customer(self, customer_id: str) -> dict[str, Any]:
        logger.debug("CrmTool.get_customer | customer_id=%s", customer_id)
        customer = self._crm_repo.find_customer_by_id(customer_id)
        if customer is None:
            raise CustomerNotFoundError(f"Customer '{customer_id}' not found.")
        return customer

    async def create_case(
        self,
        customer_id: str,
        order_id: str,
        reason: str,
        amount_inr: float | None,
        trace_id: str,
        priority: str = "medium",
    ) -> dict[str, Any]:
        logger.info(
            "CrmTool.create_case | customer=%s | order=%s | "
            "amount=%.2f | priority=%s | trace=%s",
            customer_id, order_id, amount_inr or 0.0, priority, trace_id,
        )
        self._assert_customer_exists(customer_id)
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
            "CrmTool.create_case SUCCESS | case_id=%s | customer=%s | order=%s | trace=%s",
            case["case_id"], customer_id, order_id, trace_id,
        )
        return case

    async def get_cases(
        self,
        customer_id: str,
        status_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        logger.debug("CrmTool.get_cases | customer_id=%s | filter=%s", customer_id, status_filter)
        self._assert_customer_exists(customer_id)
        cases = self._crm_repo.find_cases_by_customer(customer_id)
        if status_filter:
            cases = [c for c in cases if c.get("status") == status_filter]
        logger.debug("CrmTool.get_cases | customer=%s | found=%d", customer_id, len(cases))
        return cases
