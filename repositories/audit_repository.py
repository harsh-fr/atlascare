import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from utils.file_ops import atomic_json_write

logger = logging.getLogger(__name__)

_DEFAULT_DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "order_audit_log.json"
)


class AuditRepository:
    """Append-only event log for all support actions taken on orders.

    Each event records: who (customer_id), what order, what action,
    when, and action-specific detail. Never mutates existing entries.
    """

    def __init__(self, data_path: str | None = None) -> None:
        self._path = os.path.abspath(
            data_path or os.getenv("AUDIT_LOG_PATH", _DEFAULT_DATA_PATH)
        )
        self._events: list[dict[str, Any]] = []
        self._load()
        logger.debug(
            "AuditRepository loaded | path=%s | events=%d",
            self._path, len(self._events),
        )

    def append(
        self,
        customer_id: str,
        order_id: str,
        action: str,
        data: dict[str, Any],
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "event_id":   f"EVT-{uuid.uuid4().hex[:8].upper()}",
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            "customer_id": customer_id,
            "order_id":   order_id,
            "action":     action,
            "data":       data,
        }
        self._events.append(event)
        self._flush()
        logger.info(
            "AuditRepository.append | event_id=%s | customer=%s | order=%s | action=%s",
            event["event_id"], customer_id, order_id, action,
        )
        return event

    def find_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        return [e for e in self._events if e.get("customer_id") == customer_id]

    def find_by_order(self, order_id: str) -> list[dict[str, Any]]:
        return [e for e in self._events if e.get("order_id") == order_id]

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._events)

    def _load(self) -> None:
        if not os.path.exists(self._path):
            logger.debug("Audit log not found at '%s' — starting empty.", self._path)
            self._events = []
            return
        with open(self._path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self._events = raw.get("events", [])

    def _flush(self) -> None:
        atomic_json_write(self._path, {"events": self._events})
        logger.debug("AuditRepository flushed | path=%s | total=%d", self._path, len(self._events))
