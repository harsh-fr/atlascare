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
import base64
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

import requests as _requests
import gradio as gr
from dotenv import load_dotenv

from repositories.order_repository import OrderRepository
from repositories.audit_repository import AuditRepository

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ADMIN_PORT      = int(os.getenv("ADMIN_GRADIO_PORT", "7861"))
ADMIN_HOST      = os.getenv("ADMIN_GRADIO_HOST", "127.0.0.1")
API_PORT        = os.getenv("PORT", "8000")
API_BASE        = f"http://127.0.0.1:{API_PORT}"
DATA_DIR        = Path(os.getenv("ORDERS_DATA_PATH", "./data/orders.json")).parent
CRM_PATH        = DATA_DIR / "crm_cases.json"
REFUNDS_PATH    = DATA_DIR / "refunds.json"
ORDERS_PATH     = DATA_DIR / "orders.json"

# ---------------------------------------------------------------------------
# Trace fetcher — polls the FastAPI /admin/traces endpoint
# ---------------------------------------------------------------------------

def get_traces() -> list[dict]:
    """Fetch traces from the running FastAPI server."""
    try:
        resp = _requests.get(f"{API_BASE}/admin/traces", timeout=3)
        return resp.json() if resp.ok else []
    except Exception:
        return []


def _fetch_kpis_from_api() -> dict | None:
    """Fetch pre-computed KPIs from the FastAPI server."""
    try:
        resp = _requests.get(f"{API_BASE}/admin/kpis", timeout=3)
        return resp.json() if resp.ok else None
    except Exception:
        return None


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
    # Traffic/latency KPIs come from the FastAPI trace store via API
    api_kpis = _fetch_kpis_from_api() or {}

    # CRM/refund KPIs come from the JSON files (source of truth)
    cases   = _load_cases()
    refunds = _load_refunds()
    orders  = _load_orders()

    open_cases       = sum(1 for c in cases if c.get("status") == "open")
    high_pri_cases   = sum(1 for c in cases if c.get("priority") == "high")
    cases_with_trace = sum(1 for c in cases if c.get("trace_id"))
    total_refunds    = len(refunds)
    refund_amount    = sum(r.get("amount_inr", 0) for r in refunds)

    return {
        "total_requests":   api_kpis.get("total_requests", 0),
        "escalated":        api_kpis.get("escalated", 0),
        "escalation_rate":  api_kpis.get("escalation_rate", "0%"),
        "guardrail_hits":   api_kpis.get("guardrail_hits", 0),
        "ownership_denied": api_kpis.get("ownership_denied", 0),
        "avg_latency_ms":   api_kpis.get("avg_latency_ms", 0),
        "p99_latency_ms":   api_kpis.get("p99_latency_ms", 0),
        "sla_breaches":     api_kpis.get("sla_breaches", 0),
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
            f"`{tc.get('action','?')}` -> {tc.get('status','?')}"
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
# Orders & Audit Trace helpers (direct repo access)
# ---------------------------------------------------------------------------

_STATUS_EMOJI = {
    "placed":     "🟡 Placed",
    "processing": "🔵 Processing",
    "shipped":    "🚚 Shipped",
    "delivered":  "✅ Delivered",
    "cancelled":  "❌ Cancelled",
}
_METHOD_LABELS = {
    "HDFC_CREDIT":    "HDFC Credit",
    "ICICI_DEBIT":    "ICICI Debit",
    "SBI_NETBANKING": "SBI NetBanking",
    "UPI":            "UPI",
    "COD":            "COD",
}
_ACTION_LABELS = {
    "item_cancelled":     "🗑 Item Cancelled",
    "refund_processed":   "💸 Refund",
    "address_updated":    "📍 Address Updated",
    "escalation_created": "🚨 Escalation",
}


def _fmt_inr(amount) -> str:
    try:
        return f"₹{float(amount):,.0f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ts(ts: str) -> str:
    try:
        return datetime.fromisoformat(ts).strftime("%d %b %Y  %H:%M")
    except Exception:
        return ts or "—"


def _audit_row_detail(action: str, data: dict) -> str:
    if action == "item_cancelled":
        line = f"{data.get('item_name','?')} × {data.get('quantity',1)} @ {_fmt_inr(data.get('unit_price',0))}"
        if data.get("refund_id"):
            line += f"  |  Refund {_fmt_inr(data.get('refund_amount',0))} → {data.get('refund_method','?')}"
        return line
    if action == "refund_processed":
        note = " (escalated)" if data.get("escalated") else ""
        case = f"  |  Case {data['case_id']}" if data.get("case_id") else ""
        return f"{_fmt_inr(data.get('amount_inr',0))} → {data.get('method','?')}{note}{case}"
    if action == "address_updated":
        addr  = data.get("new_address") or {}
        label = f" [{data['address_label']}]" if data.get("address_label") else ""
        return f"{addr.get('line1','')}, {addr.get('city','')} {addr.get('pincode','')}{label}"
    if action == "escalation_created":
        reason  = data.get("reason", "")
        snippet = reason[:70] + ("…" if len(reason) > 70 else "")
        return f"Case {data.get('case_id','?')}  |  {snippet}"
    return str(data)


def _customer_choices() -> list[str]:
    customers = sorted({
        o.get("customer_id", "")
        for o in OrderRepository().list_all()
        if o.get("customer_id")
    })
    return ["All"] + customers


def load_orders(customer_filter: str = "All", status_filter: str = "All") -> pd.DataFrame:
    audit_counts: dict[str, int] = {}
    for evt in AuditRepository().list_all():
        oid = evt.get("order_id", "")
        audit_counts[oid] = audit_counts.get(oid, 0) + 1

    rows = []
    for o in OrderRepository().list_all():
        cid    = o.get("customer_id", "")
        status = o.get("status", "")
        if customer_filter != "All" and cid != customer_filter:
            continue
        if status_filter != "All" and status != status_filter:
            continue
        items    = o.get("items", [])
        n_active = sum(1 for i in items if i.get("status") == "active")
        n_cancel = sum(1 for i in items if i.get("status") == "cancelled")
        oid      = o.get("order_id", "")
        rows.append({
            "Order ID":      oid,
            "Customer":      cid,
            "Status":        _STATUS_EMOJI.get(status, status),
            "Created":       _fmt_ts(o.get("created_at", "")),
            "Est. Delivery": o.get("estimated_delivery", "—"),
            "Items":         f"{n_active} active" + (f", {n_cancel} cancelled" if n_cancel else ""),
            "Total":         _fmt_inr(o.get("total_amount", 0)),
            "Payment":       _METHOD_LABELS.get(o.get("payment_method", ""), o.get("payment_method", "")),
            "Agent Actions": f"🔔 {audit_counts[oid]}" if audit_counts.get(oid) else "—",
        })

    if not rows:
        return pd.DataFrame(columns=[
            "Order ID", "Customer", "Status", "Created", "Est. Delivery",
            "Items", "Total", "Payment", "Agent Actions",
        ])
    return pd.DataFrame(rows).sort_values("Created", ascending=False).reset_index(drop=True)


def load_audit_table(customer_filter: str = "All", action_filter: str = "All") -> pd.DataFrame:
    rows = []
    for evt in AuditRepository().list_all():
        cid    = evt.get("customer_id", "")
        action = evt.get("action", "")
        if customer_filter != "All" and cid != customer_filter:
            continue
        if action_filter != "All" and action != action_filter:
            continue
        rows.append({
            "Timestamp": _fmt_ts(evt.get("timestamp", "")),
            "Customer":  cid,
            "Order ID":  evt.get("order_id", ""),
            "Action":    _ACTION_LABELS.get(action, action),
            "Details":   _audit_row_detail(action, evt.get("data", {})),
            "Event ID":  evt.get("event_id", ""),
        })

    if not rows:
        return pd.DataFrame(columns=["Timestamp", "Customer", "Order ID", "Action", "Details", "Event ID"])
    return pd.DataFrame(rows).sort_values("Timestamp", ascending=False).reset_index(drop=True)


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

        # ── Agent Graph ───────────────────────────────────────────────────
        with gr.Tab("🕸️ Agent Graph"):
            def _graph_html() -> str:
                try:
                    from agent.graph import build_graph
                    png_bytes = build_graph().get_graph().draw_mermaid_png()
                    b64 = base64.b64encode(png_bytes).decode()
                    return (
                        '<div style="background:#1e1e2e; padding:16px; border-radius:8px; overflow:auto;">'
                        f'<img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;"/>'
                        '</div>'
                    )
                except Exception:
                    fallback = (
                        "graph TD\n"
                        "    A([START]) --> B[pre_guardrail]\n"
                        "    B -->|blocked| Z([END])\n"
                        "    B -->|allow| C[tool_agent]\n"
                        "    C -->|tool calls| D[tool_executor]\n"
                        "    C -->|no tools| E[post_guardrail]\n"
                        "    D --> E\n"
                        "    E -->|blocked| Z\n"
                        "    E -->|allow| F[responder]\n"
                        "    F --> Z"
                    )
                    return (
                        '<div style="background:#1e1e2e; color:#cdd6f4; padding:16px; border-radius:8px;">'
                        '<p style="color:#f38ba8; margin-top:0;">⚠️ Graph image unavailable — showing Mermaid source</p>'
                        f'<pre style="font-size:12px; white-space:pre-wrap;">{fallback}</pre>'
                        '</div>'
                    )

            graph_display = gr.HTML(_graph_html())
            gr.Button("🔄 Refresh Graph").click(fn=_graph_html, outputs=graph_display)

        # ── Orders ────────────────────────────────────────────────────────
        with gr.Tab("📦 Orders"):
            with gr.Row():
                o_customer_dd = gr.Dropdown(
                    choices=_customer_choices(), value="All",
                    label="Customer", scale=2,
                )
                o_status_dd = gr.Dropdown(
                    choices=["All", "placed", "processing", "shipped",
                             "delivered", "cancelled"],
                    value="All", label="Status", scale=2,
                )
                o_refresh_btn = gr.Button("🔄 Refresh", scale=1, variant="secondary")

            o_table = gr.Dataframe(
                value=load_orders(),
                interactive=False,
                wrap=True,
                column_widths=["10%", "10%", "11%", "13%", "10%", "14%", "8%", "11%", "13%"],
            )

            o_customer_dd.change(load_orders, [o_customer_dd, o_status_dd], o_table)
            o_status_dd.change(load_orders, [o_customer_dd, o_status_dd], o_table)
            o_refresh_btn.click(load_orders, [o_customer_dd, o_status_dd], o_table)

        # ── Agent Audit Traces ─────────────────────────────────────────────
        with gr.Tab("🗒️ Audit Traces"):
            with gr.Row():
                a_customer_dd = gr.Dropdown(
                    choices=_customer_choices(), value="All",
                    label="Customer", scale=2,
                )
                a_action_dd = gr.Dropdown(
                    choices=["All", "item_cancelled", "refund_processed",
                             "address_updated", "escalation_created"],
                    value="All", label="Action", scale=2,
                )
                a_refresh_btn = gr.Button("🔄 Refresh", scale=1, variant="secondary")

            a_table = gr.Dataframe(
                value=load_audit_table(),
                interactive=False,
                wrap=True,
                column_widths=["14%", "10%", "10%", "16%", "38%", "12%"],
            )

            a_customer_dd.change(load_audit_table, [a_customer_dd, a_action_dd], a_table)
            a_action_dd.change(load_audit_table, [a_customer_dd, a_action_dd], a_table)
            a_refresh_btn.click(load_audit_table, [a_customer_dd, a_action_dd], a_table)

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
    print(f"AtlasCare Admin Dashboard -> http://{ADMIN_HOST}:{ADMIN_PORT}")
    print("Customer UI runs on port 7860. This dashboard is separate.")
    dashboard.launch(
        server_name=ADMIN_HOST,
        server_port=ADMIN_PORT,
        share=False,
    )