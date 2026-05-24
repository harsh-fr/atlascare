"""
repositories/payment_repository.py
====================================
Payment configuration and refund record persistence layer.

Responsibility
--------------
  1. Load and expose payment_config.json (gateway configuration,
     auto-refund limit, supported methods, SLA, simulated behaviour).
  2. Persist refund records to refunds.json for audit trail.

Design principles
-----------------
- Config is read-only at runtime — loaded once at construction.
- Refund records are append-only — never mutated after creation.
  This preserves a complete audit trail.
- Separate config and refund stores so the config file stays clean
  and immutable while refund history grows independently.
- Atomic writes for refund persistence.
- Returns plain dicts — no internal state leaks out.
"""

import json
import logging
import os
import tempfile
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "payment_config.json"
)

_DEFAULT_REFUNDS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "refunds.json"
)


class PaymentRepository:
    """
    JSON-backed repository for payment gateway config and refund records.

    Config  : loaded once at startup, read-only.
    Refunds : append-only audit log, flushed to disk on every write.
    """

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

        self._config:  dict[str, Any]        = {}
        self._refunds: dict[str, dict[str, Any]] = {}  # refund_id → refund

        self._load_config()
        self._load_refunds()

        logger.debug(
            "PaymentRepository loaded | config=%s | refunds=%s | "
            "refund_count=%d",
            self._config_path,
            self._refunds_path,
            len(self._refunds),
        )

    # ------------------------------------------------------------------
    # Config read operations (read-only)
    # ------------------------------------------------------------------
    def get_config(self) -> dict[str, Any]:
        """
        Return the full payment gateway configuration dict.

        Keys include: auto_refund_limit_inr, supported_methods,
                      refund_sla_days, behaviour.
        """
        return dict(self._config)

    def get_auto_refund_limit(self) -> float:
        """Return the configured auto-refund limit in INR."""
        return float(self._config.get("auto_refund_limit_inr", 25000.0))

    def get_refund_sla_days(self) -> int:
        """Return the configured refund SLA in business days."""
        return int(self._config.get("refund_sla_days", 5))

    def get_supported_methods(self) -> list[str]:
        """Return list of supported payment methods."""
        return list(self._config.get("supported_methods", []))

    def get_failure_rate(self) -> float:
        """Return simulated gateway failure rate (0.0 to 1.0)."""
        return float(
            self._config.get("behaviour", {}).get("failure_rate", 0.03)
        )

    # ------------------------------------------------------------------
    # Refund read operations
    # ------------------------------------------------------------------
    def find_refund_by_id(self, refund_id: str) -> dict[str, Any] | None:
        """Return refund record or None if not found."""
        refund = self._refunds.get(refund_id)
        return dict(refund) if refund is not None else None

    def find_refunds_by_order(self, order_id: str) -> list[dict[str, Any]]:
        """Return all refund records for a given order_id."""
        return [
            dict(r) for r in self._refunds.values()
            if r.get("order_id") == order_id
        ]

    def find_refunds_by_customer(self, customer_id: str) -> list[dict[str, Any]]:
        """Return all refund records for a given customer_id, newest first."""
        refunds = [
            dict(r) for r in self._refunds.values()
            if r.get("customer_id") == customer_id
        ]
        return sorted(
            refunds,
            key=lambda r: r.get("created_at", ""),
            reverse=True,
        )

    def list_all_refunds(self) -> list[dict[str, Any]]:
        """Return all refund records."""
        return [dict(r) for r in self._refunds.values()]

    # ------------------------------------------------------------------
    # Refund write operations (append-only)
    # ------------------------------------------------------------------
    def save_refund(self, refund: dict[str, Any]) -> None:
        """
        Append a new refund record to the store and flush to disk.

        Refund records are immutable once created — this method will
        raise ValueError if a record with the same refund_id already
        exists, enforcing the append-only contract.

        Parameters
        ----------
        refund : full refund dict — must contain 'refund_id' key.

        Raises
        ------
        ValueError  if 'refund_id' missing or already exists.
        """
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
            "PaymentRepository.save_refund | refund_id=%s | order=%s | "
            "amount=%.2f",
            refund_id,
            refund.get("order_id"),
            refund.get("amount_inr", 0.0),
        )

    # ------------------------------------------------------------------
    # Private — load
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        """Load payment_config.json. Uses safe defaults if file missing."""
        if not os.path.exists(self._config_path):
            logger.warning(
                "Payment config not found at '%s' — using defaults.",
                self._config_path,
            )
            self._config = {
                "auto_refund_limit_inr": 25000.0,
                "supported_methods": [
                    "HDFC_CREDIT", "ICICI_DEBIT",
                    "SBI_NETBANKING", "UPI", "original",
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
        """Load existing refund records. Starts empty if file missing."""
        if not os.path.exists(self._refunds_path):
            logger.debug(
                "Refunds file not found at '%s' — starting empty.",
                self._refunds_path,
            )
            self._refunds = {}
            return

        with open(self._refunds_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        refunds_list: list[dict] = raw.get("refunds", [])
        self._refunds = {r["refund_id"]: r for r in refunds_list}

    # ------------------------------------------------------------------
    # Private — flush
    # ------------------------------------------------------------------
    def _flush_refunds(self) -> None:
        """
        Atomically write refund records to disk.
        Uses write-to-temp + os.replace() pattern.
        """
        os.makedirs(os.path.dirname(self._refunds_path), exist_ok=True)

        payload = {"refunds": list(self._refunds.values())}

        dir_name = os.path.dirname(self._refunds_path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=dir_name,
            delete=False,
            suffix=".tmp",
        ) as tmp:
            json.dump(payload, tmp, indent=2, ensure_ascii=False)
            tmp_path = tmp.name

        os.replace(tmp_path, self._refunds_path)
        logger.debug("PaymentRepository flushed | path=%s", self._refunds_path)