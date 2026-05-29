import json
import logging
import os
import threading
from typing import Any

from utils.file_ops import atomic_json_write, sort_by_recency

logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "orders.json")


class OrderRepository:
    def __init__(self, data_path: str | None = None) -> None:
        self._path = os.path.abspath(
            data_path or os.getenv("ORDERS_DATA_PATH", _DEFAULT_DATA_PATH)
        )
        self._orders: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load()
        logger.debug("OrderRepository loaded | path=%s | count=%d", self._path, len(self._orders))

    def find_by_id(self, order_id: str) -> dict[str, Any] | None:
        order = self._orders.get(order_id.strip().upper())
        return dict(order) if order is not None else None

    def find_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        orders = [
            dict(o) for o in self._orders.values()
            if o.get("customer_id") == customer_id
        ]
        return sort_by_recency(orders)

    def list_all(self) -> list[dict[str, Any]]:
        return [dict(o) for o in self._orders.values()]

    def exists(self, order_id: str) -> bool:
        return order_id in self._orders

    def save(self, order: dict[str, Any]) -> None:
        order_id = order.get("order_id")
        if not order_id:
            raise ValueError("Cannot save order without 'order_id'.")
        order_id = order_id.upper()
        order["order_id"] = order_id
        with self._lock:
            self._orders[order_id] = order
            self._flush()
        logger.debug("OrderRepository.save | order_id=%s", order_id)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            logger.warning("Orders data file not found at '%s' — starting empty.", self._path)
            self._orders = {}
            return
        with open(self._path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        orders_list: list[dict] = raw.get("orders", [])
        self._orders = {o["order_id"].upper(): o for o in orders_list}

    def _flush(self) -> None:
        atomic_json_write(self._path, {"orders": list(self._orders.values())})
        logger.debug("OrderRepository flushed | path=%s", self._path)
