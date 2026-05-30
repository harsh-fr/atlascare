import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

class Tracer:
    def __init__(self, session_id: str) -> None:
        self.trace_id:    str = self._generate_trace_id()
        self.session_id:  str = session_id
        self.customer_id: str = ""
        self.latency_ms:  int = 0
        self.tool_calls: list[dict[str, Any]] = []
        self._guardrail_events: list[dict[str, Any]] = []

        logger.debug(
            "Tracer created | trace_id=%s | session_id=%s",
            self.trace_id,
            self.session_id,
        )
    @staticmethod
    def _generate_trace_id() -> str:
        return f"trc-{uuid.uuid4().hex[:12]}"
        
    def set_customer_id(self, customer_id: str) -> None:
        self.customer_id = customer_id

    def set_latency(self, latency_ms: int) -> None:
        self.latency_ms = latency_ms

    def record_tool_call(
        self,
        tool: str,
        action: str,
        status: str,
        meta: dict[str, Any] | None = None,
    ) -> None:

        record: dict[str, Any] = {
            "tool":       tool,
            "action":     action,
            "status":     status,
            "latency_ms": (meta or {}).get("latency_ms", 0),
            "meta":       meta or {},
        }
        self.tool_calls.append(record)
        logger.debug(
            "Trace tool_call | trace=%s | tool=%s | action=%s | status=%s",
            self.trace_id,
            tool,
            action,
            status,
        )

    def record_guardrail_trigger(
        self,
        rule_id: str,
        phase: str,
        reason: str,
    ) -> None:

        event = {
            "rule_id": rule_id,
            "phase":   phase,
            "reason":  reason,
        }
        self._guardrail_events.append(event)
        self.record_tool_call(
            tool="guardrails",
            action=f"{phase}_check",
            status="guardrail_blocked",
            meta={"rule_id": rule_id, "reason": reason},
        )

        logger.warning(
            "Guardrail triggered | trace=%s | rule=%s | phase=%s | reason=%s",
            self.trace_id,
            rule_id,
            phase,
            reason,
        )

    def get_guardrail_events(self) -> list[dict[str, Any]]:
        return list(self._guardrail_events)

    def had_guardrail_trigger(self) -> bool:
        return len(self._guardrail_events) > 0

    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id":   self.trace_id,
            "session_id": self.session_id,
            "latency_ms": self.latency_ms,
            "tool_calls": self.tool_calls,
        }