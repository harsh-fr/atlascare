"""
admin_dashboard.py
==================
AtlasCare Admin Dashboard — KPIs, Traces, and Logs.

Run on a separate port from the customer UI.
Customer UI has NO access to this dashboard.

Usage:
    python admin_dashboard.py

Runs on http://127.0.0.1:7861 by default.
Customer Gradio UI runs on 7860.
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, deque
from threading import Lock

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ADMIN_PORT      = int(os.getenv("ADMIN_GRADIO_PORT", "7861"))
ADMIN_HOST      = os.getenv("ADMIN_GRADIO_HOST", "127.0.0.1")
DATA_DIR        = Path(os.getenv("ORDERS_DATA_PATH", "./data/orders.json")).parent
CRM_PATH        = DATA_DIR / "crm_cases.json"
REFUNDS_PATH    = DATA_DIR / "refunds.json"
ORDERS_PATH     = DATA_DIR / "orders.json"

# In-memory trace store (ring buffer — last 500 traces)
_trace_store: deque = deque(maxlen=500)
_trace_lock  = Lock()

# ---------------------------------------------------------------------------
# Trace ingestion — called by the agent at runtime
# ---------------------------------------------------------------------------

def record_trace(trace: dict):
    """Called by the agent to push a trace into the admin store."""
    with _trace_lock:
        _trace_store.append({
            **trace,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })


def get_traces() -> list[dict]:
    with _trace_lock:
        return list(reversed(_trace_store))


# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _load_cases() -> list[dict]:
    return _read_json(CRM_PATH).get("cases", [])


def _load_refunds() -> list[dict]:
    return _read_json(REFUNDS_PATH).get("refunds", [])


def _load_orders() -> list[dict]:
    return _read_json(ORDERS_PATH).get("orders", [])


# ---------------------------------------------------------------------------
# KPI calculations
# ---------------------------------------------------------------------------

def _compute_kpis() -> dict:
    traces  = get_traces()
    cases   = _load_cases()
    refunds = _load_refunds()
    orders  = _load_orders()

    total_requests   = len(traces)
    escalated        = sum(1 for t in traces if t.get("escalated"))
    guardrail_hits   = sum(
        1 for t in traces
        if any(tc.get("tool") == "guardrails" for tc in t.get("tool_calls", []))
    )
    ownership_denied = sum(
        1 for t in traces
        if any(tc.get("status") == "ownership_denied" for tc in t.get("tool_calls", []))
    )
    latencies = [t["latency_ms"] for t in traces if t.get("latency_ms")]
    avg_latency = int(sum(latencies) / len(latencies)) if latencies else 0
    p99_latency = int(sorted(latencies)[int(len(latencies) * 0.99)]) if len(latencies) > 10 else avg_latency
    breached_sla = sum(1 for l in latencies if l > 3000)

    open_cases     = sum(1 for c in cases if c.get("status") == "open")
    high_pri_cases = sum(1 for c in cases if c.get("priority") == "high")
    cases_with_trace = sum(1 for c in cases if c.get("trace_id"))

    total_refunds    = len(refunds)
    refund_amount    = sum(r.get("amount_inr", 0) for r in refunds)

    return {
        "total_requests":    total_requests,
        "escalated":         escalated,
        "escalation_rate":   f"{(escalated/total_requests*100):.1f}%" if total_requests else "0%",
        "guardrail_hits":    guardrail_hits,
        "ownership_denied":  ownership_denied,
        "avg_latency_ms":    avg_latency,
        "p99_latency_ms":    p99_latency,
        "sla_breaches":      breached_sla,
        "open_cases":        open_cases,
        "high_pri_cases":    high_pri_cases,
        "cases_with_trace":  cases_with_trace,
        "total_cases":       len(cases),
        "total_refunds":     total_refunds,
        "total_refund_inr":  f"₹{refund_amount:,.2f}",
        "total_orders":      len(orders),
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _fmt_kpis() -> str:
    k = _compute_kpis()
    return f"""
## 📊 Live KPIs

| Category | Metric | Value |
|----------|--------|-------|
| **Traffic** | Total requests (session) | {k['total_requests']} |
| **Traffic** | Avg latency | {k['avg_latency_ms']} ms |
| **Traffic** | P99 latency | {k['p99_latency_ms']} ms |
| **Traffic** | SLA breaches (>3s) | {k['sla_breaches']} |
| **Safety** | Guardrail triggers | {k['guardrail_hits']} |
| **Safety** | Ownership denials | {k['ownership_denied']} |
| **Safety** | Escalation rate | {k['escalation_rate']} |
| **CRM** | Open cases | {k['open_cases']} |
| **CRM** | High priority cases | {k['high_pri_cases']} |
| **CRM** | Cases with trace | {k['cases_with_trace']} / {k['total_cases']} |
| **Payments** | Total refunds | {k['total_refunds']} |
| **Payments** | Total refund value | {k['total_refund_inr']} |

*Last updated: {datetime.now().strftime('%H:%M:%S')}*
""".strip()


def _fmt_traces(limit: int = 20) -> str:
    traces = get_traces()[:limit]
    if not traces:
        return "_No traces yet. Make some requests to the agent first._"

    lines = ["## 🔍 Recent Traces\n"]
    for t in traces:
        tool_summary = ", ".join(
            f"`{tc.get('action','?')}` → {tc.get('status','?')}"
            for tc in t.get("tool_calls", [])
        )
        guardrail = "🛡️ " if any(
            tc.get("tool") == "guardrails" for tc in t.get("tool_calls", [])
        ) else ""
        escalated = "🔺 " if t.get("escalated") else ""
        lines.append(
            f"**{guardrail}{escalated}{t.get('trace_id','?')}**  "
            f"| session: `{t.get('session_id','?')}`  "
            f"| {t.get('latency_ms','?')} ms  "
            f"| {t.get('recorded_at','')[:19]}\n"
            f"> {tool_summary if tool_summary else '_no tool calls_'}\n"
        )
    return "\n".join(lines)


def _fmt_cases() -> str:
    cases = _load_cases()
    if not cases:
        return "_No CRM cases found._"

    lines = ["## 🗂️ CRM Escalation Cases\n"]
    for c in sorted(cases, key=lambda x: x.get("created_at",""), reverse=True)[:30]:
        status_icon = {"open": "🔴", "in_progress": "🟡", "resolved": "🟢", "closed": "⚫"}.get(c.get("status",""), "❓")
        pri_icon    = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(c.get("priority",""), "")
        amount      = f"₹{c['amount_inr']:,.0f}" if c.get("amount_inr") else "N/A"
        trace_link  = f"`{c.get('trace_id','—')}`"
        lines.append(
            f"{status_icon} **{c.get('case_id','?')}** {pri_icon}  "
            f"| Order: `{c.get('order_id','?')}`  "
            f"| Customer: `{c.get('customer_id','?')}`  "
            f"| Amount: {amount}  "
            f"| Trace: {trace_link}  "
            f"| {c.get('created_at','')[:10]}\n"
        )
    return "\n".join(lines)


def _fmt_refunds() -> str:
    refunds = _load_refunds()
    if not refunds:
        return "_No refunds processed yet._"

    lines = ["## 💳 Refund Audit Log\n"]
    total = 0.0
    for r in sorted(refunds, key=lambda x: x.get("created_at",""), reverse=True)[:30]:
        amt = r.get("amount_inr", 0)
        total += amt
        lines.append(
            f"**{r.get('refund_id','?')}**  "
            f"| Order: `{r.get('order_id','?')}`  "
            f"| ₹{amt:,.2f}  "
            f"| {r.get('method','?')}  "
            f"| {r.get('status','?')}  "
            f"| {r.get('created_at','')[:19]}\n"
        )
    lines.insert(1, f"**Total shown:** ₹{total:,.2f}\n")
    return "\n".join(lines)


def _fmt_trace_detail(trace_id: str) -> str:
    traces = get_traces()
    match  = next((t for t in traces if t.get("trace_id") == trace_id.strip()), None)
    if not match:
        return f"_Trace `{trace_id}` not found in session buffer._"
    return f"```json\n{json.dumps(match, indent=2, default=str)}\n```"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def refresh_all():
    return _fmt_kpis(), _fmt_traces(), _fmt_cases(), _fmt_refunds()


with gr.Blocks(title="AtlasCare Admin", theme=gr.themes.Soft()) as dashboard:

    gr.Markdown("# 🛡️ AtlasCare Admin Dashboard")
    gr.Markdown(
        "> **Internal use only.** This dashboard is not accessible to customers. "
        "Run on a separate port (default 7861)."
    )

    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh All", variant="primary", scale=1)
        # Fix applied here: Wrapped in a Column to allow scaling.
        with gr.Column(scale=4):
            gr.Markdown("Auto-refreshes when you click Refresh All.")

    with gr.Tabs():

        # ── KPIs ──────────────────────────────────────────────────────────
        with gr.Tab("📊 KPIs"):
            kpi_display = gr.Markdown(_fmt_kpis())

        # ── Traces ────────────────────────────────────────────────────────
        with gr.Tab("🔍 Traces"):
            trace_limit = gr.Slider(5, 100, value=20, step=5, label="Show last N traces")
            trace_display = gr.Markdown(_fmt_traces())

            gr.Markdown("### Trace Detail")
            trace_id_input = gr.Textbox(
                label="Enter trace_id (e.g. trc-a1b2c3d4e5f6)",
                placeholder="trc-..."
            )
            trace_detail_btn  = gr.Button("🔎 Fetch Detail")
            trace_detail_out  = gr.Code(language="json", label="Full trace JSON")

            def fetch_trace_detail(tid):
                return _fmt_trace_detail(tid)

            trace_detail_btn.click(
                fn=fetch_trace_detail,
                inputs=trace_id_input,
                outputs=trace_detail_out,
            )

            def refresh_traces(limit):
                return _fmt_traces(int(limit))

            trace_limit.change(
                fn=refresh_traces,
                inputs=trace_limit,
                outputs=trace_display,
            )

        # ── CRM Cases ─────────────────────────────────────────────────────
        with gr.Tab("🗂️ CRM Cases"):
            cases_display = gr.Markdown(_fmt_cases())

        # ── Refunds ───────────────────────────────────────────────────────
        with gr.Tab("💳 Refunds"):
            refunds_display = gr.Markdown(_fmt_refunds())

        # ── Raw JSON viewer ───────────────────────────────────────────────
        with gr.Tab("📁 Raw Data"):
            file_choice = gr.Radio(
                ["crm_cases.json", "refunds.json", "orders.json"],
                label="Select file",
                value="crm_cases.json",
            )
            raw_display = gr.Code(language="json", label="File contents")

            def load_raw(filename):
                path = DATA_DIR / filename
                try:
                    return path.read_text(encoding="utf-8")
                except Exception as e:
                    return f"Error reading {filename}: {e}"

            file_choice.change(
                fn=load_raw,
                inputs=file_choice,
                outputs=raw_display,
            )
            gr.Button("📖 Load File").click(
                fn=load_raw,
                inputs=file_choice,
                outputs=raw_display,
            )

    # Refresh all tabs
    refresh_btn.click(
        fn=refresh_all,
        outputs=[kpi_display, trace_display, cases_display, refunds_display],
    )


if __name__ == "__main__":
    print(f"AtlasCare Admin Dashboard → http://{ADMIN_HOST}:{ADMIN_PORT}")
    print("Customer UI runs on port 7860. This dashboard is separate.")
    dashboard.launch(
        server_name=ADMIN_HOST,
        server_port=ADMIN_PORT,
        share=False,
    )