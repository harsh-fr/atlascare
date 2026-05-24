"""
repositories/order_repository.py
=================================
Order data persistence layer.

Responsibility
--------------
  Owns all read/write access to orders.json (the OMS data store).
  Provides a clean typed interface so the rest of the codebase
  never touches raw JSON files directly.

Design principles
-----------------
- Single source of truth for order data access patterns.
- JSON file is loaded once at construction and kept in memory;
  writes flush back to disk atomically (write-to-temp + rename).
- All public methods return plain dicts — no leaking of internal
  storage representation.
- Thread safety: for this assignment (single-process FastAPI with
  async routes) a simple in-memory dict is sufficient. A production
  system would replace this with a database-backed repository
  implementing the same interface.
"""

import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

# Default data file path — overridable via environment variable
_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "orders.json"
)


class OrderRepository:
    """
    JSON-backed repository for order data.

    All methods operate on the in-memory index built at construction.
    Mutations are flushed to disk after every write.
    """

    def __init__(self, data_path: str | None = None) -> None:
        self._path = os.path.abspath(
            data_path or os.getenv("ORDERS_DATA_PATH", _DEFAULT_DATA_PATH)
        )
        self._orders: dict[str, dict[str, Any]] = {}
        self._load()
        logger.debug(
            "OrderRepository loaded | path=%s | count=%d",
            self._path,
            len(self._orders),
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def find_by_id(self, order_id: str) -> dict[str, Any] | None:
        """
        Return the order dict for the given order_id, or None if not found.
        Lookup is case-insensitive — "ord-78321" resolves to "ORD-78321".
        Returns a shallow copy to prevent accidental mutation of the index.
        """
        order = self._orders.get(order_id.strip().upper())
        return dict(order) if order is not None else None

    def find_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """
        Return all orders belonging to a customer, newest first.
        """
        orders = [
            dict(o) for o in self._orders.values()
            if o.get("customer_id") == customer_id
        ]
        return sorted(
            orders,
            key=lambda o: o.get("created_at", ""),
            reverse=True,
        )

    def list_all(self) -> list[dict[str, Any]]:
        """Return all orders as a list of dicts."""
        return [dict(o) for o in self._orders.values()]

    def exists(self, order_id: str) -> bool:
        """Return True if the order_id exists in the store."""
        return order_id in self._orders

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def save(self, order: dict[str, Any]) -> None:
        """
        Upsert an order into the store and flush to disk.

        Parameters
        ----------
        order : full order dict — must contain 'order_id' key.

        Raises
        ------
        ValueError  if 'order_id' is missing from the dict.
        """
        order_id = order.get("order_id")
        if not order_id:
            raise ValueError("Cannot save order without 'order_id'.")

        order_id = order_id.upper()
        order["order_id"] = order_id          # keep record consistent
        self._orders[order_id] = order
        self._flush()
        logger.debug("OrderRepository.save | order_id=%s", order_id)

    # ------------------------------------------------------------------
    # Private — load / flush
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """
        Load orders from the JSON file into the in-memory index.
        Creates an empty store if the file does not exist.
        """
        if not os.path.exists(self._path):
            logger.warning(
                "Orders data file not found at '%s' — starting empty.",
                self._path,
            )
            self._orders = {}
            return

        with open(self._path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        orders_list: list[dict] = raw.get("orders", [])
        self._orders = {o["order_id"].upper(): o for o in orders_list}

    def _flush(self) -> None:
        """
        Write the current in-memory state back to the JSON file.

        Uses a write-to-temp-then-rename pattern to prevent data
        corruption if the process is interrupted mid-write.
        """
        os.makedirs(os.path.dirname(self._path), exist_ok=True)

        payload = {"orders": list(self._orders.values())}

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
        logger.debug("OrderRepository flushed | path=%s", self._path)