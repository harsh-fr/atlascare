"""
repositories/crm_repository.py
================================
CRM data persistence layer.

Responsibility
--------------
  Owns all read/write access to crm_cases.json (customers + cases).
  Provides a clean typed interface so the rest of the codebase
  never touches raw JSON files directly.

Design principles
-----------------
- Separate in-memory indexes for customers and cases for O(1) lookup.
- Atomic disk writes using write-to-temp + rename pattern.
- Returns plain dicts (shallow copies) — no internal state leaks out.
- Path overridable via env var for test isolation.
"""

import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "crm_cases.json"
)


class CrmRepository:
    """
    JSON-backed repository for customer profiles and CRM cases.

    Maintains two independent in-memory indexes:
      _customers : customer_id  → customer dict
      _cases     : case_id      → case dict
    """

    def __init__(self, data_path: str | None = None) -> None:
        self._path = os.path.abspath(
            data_path or os.getenv("CRM_DATA_PATH", _DEFAULT_DATA_PATH)
        )
        self._customers: dict[str, dict[str, Any]] = {}
        self._cases:     dict[str, dict[str, Any]] = {}
        self._load()
        logger.debug(
            "CrmRepository loaded | path=%s | customers=%d | cases=%d",
            self._path,
            len(self._customers),
            len(self._cases),
        )

    # ------------------------------------------------------------------
    # Customer read operations
    # ------------------------------------------------------------------
    def find_customer_by_id(self, customer_id: str) -> dict[str, Any] | None:
        """Return customer dict or None if not found."""
        customer = self._customers.get(customer_id)
        return dict(customer) if customer is not None else None

    def find_customer_by_email(self, email: str) -> dict[str, Any] | None:
        """Return customer dict matching email (case-insensitive), or None."""
        email_lower = email.lower().strip()
        for customer in self._customers.values():
            if customer.get("email", "").lower() == email_lower:
                return dict(customer)
        return None

    def list_all_customers(self) -> list[dict[str, Any]]:
        """Return all customer dicts."""
        return [dict(c) for c in self._customers.values()]

    # ------------------------------------------------------------------
    # Case read operations
    # ------------------------------------------------------------------
    def find_case_by_id(self, case_id: str) -> dict[str, Any] | None:
        """Return case dict or None if not found."""
        case = self._cases.get(case_id)
        return dict(case) if case is not None else None

    def find_cases_by_customer(
        self,
        customer_id: str,
    ) -> list[dict[str, Any]]:
        """
        Return all cases for a customer, newest first.
        """
        cases = [
            dict(c) for c in self._cases.values()
            if c.get("customer_id") == customer_id
        ]
        return sorted(
            cases,
            key=lambda c: c.get("created_at", ""),
            reverse=True,
        )

    def find_cases_by_order(self, order_id: str) -> list[dict[str, Any]]:
        """Return all cases linked to a given order_id."""
        return [
            dict(c) for c in self._cases.values()
            if c.get("order_id") == order_id
        ]

    def list_all_cases(self) -> list[dict[str, Any]]:
        """Return all case dicts."""
        return [dict(c) for c in self._cases.values()]

    # ------------------------------------------------------------------
    # Case write operations
    # ------------------------------------------------------------------
    def save_case(self, case: dict[str, Any]) -> None:
        """
        Upsert a case into the store and flush to disk.

        Parameters
        ----------
        case : full case dict — must contain 'case_id' key.

        Raises
        ------
        ValueError  if 'case_id' is missing.
        """
        case_id = case.get("case_id")
        if not case_id:
            raise ValueError("Cannot save case without 'case_id'.")

        self._cases[case_id] = case
        self._flush()
        logger.debug("CrmRepository.save_case | case_id=%s", case_id)

    def save_customer(self, customer: dict[str, Any]) -> None:
        """
        Upsert a customer profile and flush to disk.

        Parameters
        ----------
        customer : full customer dict — must contain 'customer_id' key.

        Raises
        ------
        ValueError  if 'customer_id' is missing.
        """
        customer_id = customer.get("customer_id")
        if not customer_id:
            raise ValueError("Cannot save customer without 'customer_id'.")

        self._customers[customer_id] = customer
        self._flush()
        logger.debug(
            "CrmRepository.save_customer | customer_id=%s", customer_id
        )

    # ------------------------------------------------------------------
    # Private — load / flush
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """
        Load customers and cases from the JSON file.
        Creates empty indexes if the file does not exist.
        """
        if not os.path.exists(self._path):
            logger.warning(
                "CRM data file not found at '%s' — starting empty.",
                self._path,
            )
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
        """
        Atomically write current in-memory state back to the JSON file.
        Uses write-to-temp + os.replace() to prevent corruption.
        """
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

        payload = {
            "customers": list(self._customers.values()),
            "cases":     list(self._cases.values()),
        }

        dir_name = os.path.dirname(self._path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dir_name,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name

        os.replace(tmp_path, self._path)
        logger.debug("CrmRepository flushed | path=%s", self._path)