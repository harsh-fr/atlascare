import json
import logging
import os
import threading
from typing import Any

from utils.file_ops import atomic_json_write, sort_by_recency

logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "crm_cases.json")


class CrmRepository:
    def __init__(self, data_path: str | None = None) -> None:
        self._path = os.path.abspath(
            data_path or os.getenv("CRM_DATA_PATH", _DEFAULT_DATA_PATH)
        )
        self._customers: dict[str, dict[str, Any]] = {}
        self._cases:     dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()
        logger.debug(
            "CrmRepository loaded | path=%s | customers=%d | cases=%d",
            self._path, len(self._customers), len(self._cases),
        )

    def find_customer_by_id(self, customer_id: str) -> dict[str, Any] | None:
        customer = self._customers.get(customer_id)
        return dict(customer) if customer is not None else None

    def find_customer_by_email(self, email: str) -> dict[str, Any] | None:
        email_lower = email.lower().strip()
        for customer in self._customers.values():
            if customer.get("email", "").lower() == email_lower:
                return dict(customer)
        return None

    def list_all_customers(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self._customers.values()]

    def find_case_by_id(self, case_id: str) -> dict[str, Any] | None:
        case = self._cases.get(case_id)
        return dict(case) if case is not None else None

    def find_cases_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        cases = [
            dict(c) for c in self._cases.values()
            if c.get("customer_id") == customer_id
        ]
        return sort_by_recency(cases)

    def find_cases_by_order(self, order_id: str) -> list[dict[str, Any]]:
        return [dict(c) for c in self._cases.values() if c.get("order_id") == order_id]

    def list_all_cases(self) -> list[dict[str, Any]]:
        return [dict(c) for c in self._cases.values()]

    def save_case(self, case: dict[str, Any]) -> None:
        case_id = case.get("case_id")
        if not case_id:
            raise ValueError("Cannot save case without 'case_id'.")
        with self._lock:
            self._cases[case_id] = case
            self._flush()
        logger.debug("CrmRepository.save_case | case_id=%s", case_id)

    def save_customer(self, customer: dict[str, Any]) -> None:
        customer_id = customer.get("customer_id")
        if not customer_id:
            raise ValueError("Cannot save customer without 'customer_id'.")
        with self._lock:
            self._customers[customer_id] = customer
            self._flush()
        logger.debug("CrmRepository.save_customer | customer_id=%s", customer_id)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            logger.warning("CRM data file not found at '%s' — starting empty.", self._path)
            self._customers = {}
            self._cases = {}
            return
        with open(self._path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        customers_list: list[dict] = raw.get("customers", [])
        cases_list:     list[dict] = raw.get("cases", [])
        self._customers = {c["customer_id"]: c for c in customers_list}
        self._cases     = {c["case_id"]:     c for c in cases_list}

    def _flush(self) -> None:
        atomic_json_write(self._path, {
            "customers": list(self._customers.values()),
            "cases":     list(self._cases.values()),
        })
        logger.debug("CrmRepository flushed | path=%s", self._path)
