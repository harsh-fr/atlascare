import json
import logging
import os
from typing import Any

from utils.file_ops import atomic_json_write, sort_by_recency

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "payment_config.json")
_DEFAULT_REFUNDS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "refunds.json")


class PaymentRepository:
    def __init__(
        self,
        config_path:  str | None = None,
        refunds_path: str | None = None,
    ) -> None:
        self._config_path = os.path.abspath(
            config_path or os.getenv("PAYMENT_CONFIG_PATH", _DEFAULT_CONFIG_PATH)
        )
        self._refunds_path = os.path.abspath(
            refunds_path or os.getenv("REFUNDS_DATA_PATH", _DEFAULT_REFUNDS_PATH)
        )
        self._config:  dict[str, Any]            = {}
        self._refunds: dict[str, dict[str, Any]] = {}
        self._load_config()
        self._load_refunds()
        logger.debug(
            "PaymentRepository loaded | config=%s | refunds=%s | refund_count=%d",
            self._config_path, self._refunds_path, len(self._refunds),
        )

    def get_config(self) -> dict[str, Any]:
        return dict(self._config)

    def get_auto_refund_limit(self) -> float:
        return float(self._config.get("auto_refund_limit_inr", 25000.0))

    def get_refund_sla_days(self) -> int:
        return int(self._config.get("refund_sla_days", 5))

    def get_supported_methods(self) -> list[str]:
        return list(self._config.get("supported_methods", []))

    def get_failure_rate(self) -> float:
        return float(self._config.get("behaviour", {}).get("failure_rate", 0.03))

    def find_refund_by_id(self, refund_id: str) -> dict[str, Any] | None:
        refund = self._refunds.get(refund_id)
        return dict(refund) if refund is not None else None

    def find_refunds_by_order(self, order_id: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._refunds.values() if r.get("order_id") == order_id]

    def find_refunds_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        refunds = [
            dict(r) for r in self._refunds.values()
            if r.get("customer_id") == customer_id
        ]
        return sort_by_recency(refunds)

    def list_all_refunds(self) -> list[dict[str, Any]]:
        return [dict(r) for r in self._refunds.values()]

    def save_refund(self, refund: dict[str, Any]) -> None:
        refund_id = refund.get("refund_id")
        if not refund_id:
            raise ValueError("Cannot save refund without 'refund_id'.")
        if refund_id in self._refunds:
            raise ValueError(
                f"Refund '{refund_id}' already exists. "
                "Refund records are immutable — create a new record instead."
            )
        self._refunds[refund_id] = refund
        self._flush_refunds()
        logger.debug(
            "PaymentRepository.save_refund | refund_id=%s | order=%s | amount=%.2f",
            refund_id, refund.get("order_id"), refund.get("amount_inr", 0.0),
        )

    def _load_config(self) -> None:
        if not os.path.exists(self._config_path):
            logger.warning("Payment config not found at '%s' — using defaults.", self._config_path)
            self._config = {
                "auto_refund_limit_inr": 25000.0,
                "supported_methods": [
                    "HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original",
                ],
                "refund_sla_days": 5,
                "behaviour": {
                    "failure_rate": 0.03,
                    "failure_code": "504",
                    "failure_message": "PAYMENT_GATEWAY_TIMEOUT",
                },
            }
            return
        with open(self._config_path, "r", encoding="utf-8") as fh:
            self._config = json.load(fh)

    def _load_refunds(self) -> None:
        if not os.path.exists(self._refunds_path):
            logger.debug("Refunds file not found at '%s' — starting empty.", self._refunds_path)
            self._refunds = {}
            return
        with open(self._refunds_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        refunds_list: list[dict] = raw.get("refunds", [])
        self._refunds = {r["refund_id"]: r for r in refunds_list}

    def _flush_refunds(self) -> None:
        atomic_json_write(self._refunds_path, {"refunds": list(self._refunds.values())})
        logger.debug("PaymentRepository flushed | path=%s", self._refunds_path)
