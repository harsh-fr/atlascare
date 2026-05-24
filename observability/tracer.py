"""
observability/tracer.py
========================
Per-request trace collector.

Responsibility
--------------
  Accumulates all observability data for a single request:
    - trace_id          : unique identifier generated at request start
    - session_id        : from the HTTP request
    - customer_id       : resolved from session
    - latency_ms        : set once at request completion
    - tool_calls        : ordered list of tool invocation records
    - guardrail_events  : any guardrail triggers during the request

Design principles
-----------------
- One Tracer instance per request — created in main.py, passed
  through the pipeline, read back by main.py to build the response.
- Tracer is the single mutable object in the pipeline. Everything
  else is immutable or stateless.
- trace_id format: trc-<uuid4_hex_12> — short enough to log,
  unique enough for production correlation.
- tool_calls list is the authoritative record of what happened.
  It must represent reality — no fabricated entries.
- Tracer serialises to the TraceModel shape expected by the API.
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------
class Tracer:
    """
    Mutable per-request observability accumulator.

    Created once per request in main.py and passed down through
    every layer. Each layer appends its events; main.py reads
    the final state to build the API trace payload.
    """

    def __init__(self, session_id: str) -> None:
        self.trace_id:    str = self._generate_trace_id()
        self.session_id:  str = session_id
        self.customer_id: str = ""
        self.latency_ms:  int = 0

        # Ordered list of tool call records — matches API contract
        self.tool_calls: list[dict[str, Any]] = []

        # Internal guardrail event log — not in API response but
        # available for structured logging and audit export
        self._guardrail_events: list[dict[str, Any]] = []

        logger.debug(
            "Tracer created | trace_id=%s | session_id=%s",
            self.trace_id,
            self.session_id,
        )

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------
    def set_customer_id(self, customer_id: str) -> None:
        """Record the resolved customer identity."""
        self.customer_id = customer_id

    def set_latency(self, latency_ms: int) -> None:
        """Set total end-to-end latency. Called once in main.py."""
        self.latency_ms = latency_ms

    # ------------------------------------------------------------------
    # Event recorders
    # ------------------------------------------------------------------
    def record_tool_call(
        self,
        tool: str,
        action: str,
        status: str,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Append a tool invocation record.

        Parameters
        ----------
        tool    : tool name (e.g. "oms_tool", "planner")
        action  : action performed (e.g. "get_order", "plan")
        status  : outcome ("success", "error", "ownership_denied",
                  "skipped", "guardrail_blocked")
        meta    : optional dict with extra context (latency, step index,
                  error message, prompt version, etc.)
        """
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
        """
        Record a guardrail trigger event.

        Parameters
        ----------
        rule_id : e.g. "GR-001"
        phase   : "pre" or "post"
        reason  : internal audit reason string
        """
        event = {
            "rule_id": rule_id,
            "phase":   phase,
            "reason":  reason,
        }
        self._guardrail_events.append(event)

        # Also record as a tool_call entry so it appears in the
        # API trace — evaluators can see guardrail triggers
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

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_guardrail_events(self) -> list[dict[str, Any]]:
        """Return the internal guardrail event log (for audit export)."""
        return list(self._guardrail_events)

    def had_guardrail_trigger(self) -> bool:
        """Return True if any guardrail fired during this request."""
        return len(self._guardrail_events) > 0

    def tool_call_count(self) -> int:
        """Return number of tool calls recorded."""
        return len(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise to the TraceModel-compatible dict shape.
        Used by main.py to build the API response trace payload.
        """
        return {
            "trace_id":   self.trace_id,
            "session_id": self.session_id,
            "latency_ms": self.latency_ms,
            "tool_calls": self.tool_calls,
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_trace_id() -> str:
        """
        Generate a unique trace ID.
        Format: trc-<12 hex chars>
        Example: trc-a1b2c3d4e5f6
        Short enough to include in log lines without truncation.
        """
        return f"trc-{uuid.uuid4().hex[:12]}"