"""
observability/trace_store.py
=============================
Singleton in-memory trace store.

Shared between:
  - main.py        : writes one trace per request
  - admin_dashboard.py : reads traces for KPI display

Design
------
  Module-level singleton — both main.py and admin_dashboard.py import
  get_store() and receive the same object as long as they run in the
  same Python process.

  Thread-safe ring buffer (deque maxlen=500). Oldest traces are
  automatically evicted when the buffer is full.

  All fields are plain Python primitives (str, int, bool, list, dict)
  so the store is always JSON-serialisable without extra work.
"""

import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Store implementation
# ---------------------------------------------------------------------------
class TraceStore:
    """
    Thread-safe append-only ring buffer of request traces.
    Holds the last 500 traces in memory.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._lock:   threading.Lock  = threading.Lock()
        self._buffer: deque[dict[str, Any]] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    def record(
        self,
        trace_id:          str,
        session_id:        str,
        customer_id:       str,
        message:           str,
        response:          str,
        latency_ms:        int,
        tool_calls:        list[dict[str, Any]],
        escalated:         bool = False,
        guardrail_blocked: bool = False,
        error:             bool = False,
    ) -> None:
        """
        Append a completed request trace to the buffer.

        message and response are truncated to 200 chars to keep
        memory usage bounded — full content is not needed for KPIs.
        """
        entry: dict[str, Any] = {
            "trace_id":          trace_id,
            "session_id":        session_id,
            "customer_id":       customer_id,
            "message_preview":   message[:200],
            "response_preview":  response[:200],
            "latency_ms":        latency_ms,
            "tool_calls":        tool_calls,
            "escalated":         escalated,
            "guardrail_blocked": guardrail_blocked,
            "error":             error,
            "tool_count":        len(tool_calls),
            "recorded_at":       datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self._buffer.append(entry)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_all(self) -> list[dict[str, Any]]:
        """
        Return all traces, newest first.
        Returns a snapshot copy — safe to iterate without holding the lock.
        """
        with self._lock:
            return list(reversed(self._buffer))

    def get_by_trace_id(self, trace_id: str) -> dict[str, Any] | None:
        """Return the trace matching trace_id, or None if not found."""
        with self._lock:
            for entry in self._buffer:
                if entry["trace_id"] == trace_id:
                    return dict(entry)
        return None

    def get_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Return all traces for a session, newest first."""
        with self._lock:
            matches = [
                dict(e) for e in self._buffer
                if e["session_id"] == session_id
            ]
        return list(reversed(matches))

    def count(self) -> int:
        """Total number of traces currently in the buffer."""
        with self._lock:
            return len(self._buffer)

    def clear(self) -> None:
        """
        Empty the buffer. Used in tests to reset state between runs.
        Not called in production.
        """
        with self._lock:
            self._buffer.clear()

    # ------------------------------------------------------------------
    # KPI helpers — called by admin_dashboard.py
    # ------------------------------------------------------------------
    def kpi_summary(self) -> dict[str, Any]:
        """
        Compute KPI metrics directly from the buffer.
        Returns a plain dict of JSON-safe primitives.
        """
        with self._lock:
            traces = list(self._buffer)

        total            = len(traces)
        escalated_count  = sum(1 for t in traces if t.get("escalated"))
        guardrail_count  = sum(1 for t in traces if t.get("guardrail_blocked"))
        error_count      = sum(1 for t in traces if t.get("error"))
        latencies        = [t["latency_ms"] for t in traces if t.get("latency_ms", 0) > 0]

        avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0

        sorted_lat  = sorted(latencies)
        p50_latency = sorted_lat[int(len(sorted_lat) * 0.50)] if len(sorted_lat) >= 2 else avg_latency
        p99_latency = sorted_lat[int(len(sorted_lat) * 0.99)] if len(sorted_lat) >= 2 else avg_latency
        sla_breaches = sum(1 for l in latencies if l > 3000)

        # Ownership denial count
        ownership_denied = sum(
            1 for t in traces
            if any(tc.get("status") == "ownership_denied"
                   for tc in t.get("tool_calls", []))
        )

        # Unique sessions
        unique_sessions = len({t["session_id"] for t in traces})

        return {
            "total_requests":    total,
            "unique_sessions":   unique_sessions,
            "escalated":         escalated_count,
            "escalation_rate":   f"{(escalated_count / total * 100):.1f}%" if total else "0.0%",
            "guardrail_hits":    guardrail_count,
            "ownership_denied":  ownership_denied,
            "error_count":       error_count,
            "avg_latency_ms":    avg_latency,
            "p50_latency_ms":    p50_latency,
            "p99_latency_ms":    p99_latency,
            "sla_breaches":      sla_breaches,
        }

    def __len__(self) -> int:
        return self.count()

    def __repr__(self) -> str:
        return f"TraceStore(count={self.count()})"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_instance: TraceStore = TraceStore()


def get_store() -> TraceStore:
    """
    Return the global TraceStore singleton.

    Both main.py and admin_dashboard.py call this function.
    As long as they run in the same Python process (which they do
    when launched with uvicorn), they share the same instance and
    traces written by the agent are immediately visible in the
    dashboard.
    """
    return _instance