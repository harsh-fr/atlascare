import json
import logging
import operator
import os
import re
import time
from typing import Annotated, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from openai import AsyncOpenAI

from agent.guardrails import Guardrails
from tools.oms_tool import OmsTool
from tools.crm_tool import CrmTool
from tools.payment_tool import PaymentTool
from tools.kb_tool import KbTool
from observability.tracer import Tracer

logger = logging.getLogger(__name__)

_guardrails = Guardrails()
_oms        = OmsTool()
_crm        = CrmTool()
_payment    = PaymentTool()
_kb         = KbTool()


class AtlasCareState(TypedDict):
    messages:          Annotated[list, operator.add]
    customer_id:       str
    session_id:        str
    guardrail_blocked: bool
    execution_summary: list[dict[str, Any]]
    tool_call_count:   int
    final_response:    str
    task_complete:     bool


#tools
TOOLS = [
    {"type": "function", "function": {
        "name": "get_order",
        "description": "Fetch a single order by ID. Returns status, items, tracking, and payment method.",
        "parameters": {"type": "object", "required": ["order_id"], "properties": {
            "order_id": {"type": "string", "description": "Order ID in format ORD-XXXXX (5 digits)"}
        }},
    }},
    {"type": "function", "function": {
        "name": "list_orders",
        "description": "List all orders for the authenticated customer.",
        "parameters": {"type": "object", "required": [], "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "cancel_item",
        "description": "Cancel a single line item. Only works for placed/processing orders.",
        "parameters": {"type": "object", "required": ["order_id", "line_id"], "properties": {
            "order_id": {"type": "string"},
            "line_id":  {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "process_refund",
        "description": "Process a refund. Max Rs.25,000 for autonomous processing.",
        "parameters": {"type": "object", "required": ["order_id", "amount_inr", "method"], "properties": {
            "order_id":   {"type": "string"},
            "amount_inr": {"type": "number"},
            "method":     {"type": "string",
                           "enum": ["HDFC_CREDIT", "ICICI_DEBIT", "SBI_NETBANKING", "UPI", "original"]},
        }},
    }},
    {"type": "function", "function": {
        "name": "update_address",
        "description": "Update shipping address using a saved address label.",
        "parameters": {"type": "object", "required": ["order_id", "address_label"], "properties": {
            "order_id":      {"type": "string"},
            "address_label": {"type": "string", "description": "e.g. 'home' or 'office'"},
        }},
    }},
    {"type": "function", "function": {
        "name": "create_crm_case",
        "description": "Create a support case in the CRM.",
        "parameters": {"type": "object", "required": ["order_id", "reason"], "properties": {
            "order_id":   {"type": "string"},
            "reason":     {"type": "string"},
            "amount_inr": {"type": ["number", "null"]},
        }},
    }},
    {"type": "function", "function": {
        "name": "escalate",
        "description": "Escalate to specialist team with a high-priority CRM case.",
        "parameters": {"type": "object", "required": ["order_id", "reason"], "properties": {
            "order_id":   {"type": "string"},
            "reason":     {"type": "string"},
            "amount_inr": {"type": ["number", "null"]},
        }},
    }},
    {"type": "function", "function": {
        "name": "list_cases",
        "description": "List all support cases for the authenticated customer.",
        "parameters": {"type": "object", "required": [], "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "search_kb",
        "description": "Search knowledge base for policy articles by tags.",
        "parameters": {"type": "object", "required": ["tags"], "properties": {
            "tags": {"type": "array", "items": {"type": "string"}},
        }},
    }},
]

_AGENT_SYSTEM = (
    "You are AtlasCare, a support assistant for Acme Retail Co. Be warm, patient, and genuinely helpful.\n"
    "Your name is AtlasCare — never use placeholder brackets like [Agent's Name] or [Name].\n"
    "Treat every customer message as an opportunity to assist — never be dismissive, and never imply\n"
    "the customer is repeating themselves. If they ask again, help again with the same care.\n"
    "Use the available tools to fulfill the customer's request.\n"
    "Prefer calling all needed tools in a single response when possible.\n"
    "If you call tools, do NOT write a customer reply — the response is generated from tool results.\n"
    "If no tools are needed, write a direct, helpful reply to the customer.\n"
    "For refunds, default method to 'original' unless the customer specifies otherwise.\n"
    "'Return' and 'refund' mean the same thing — always look up the order and help the customer.\n"
    "For COD (Cash on Delivery) orders: cash cannot be refunded electronically. "
    "Always ask the customer for a preferred electronic method (UPI, credit card, debit card, net banking) "
    "before calling process_refund on a COD order."
)

_RESPONSE_SYSTEM = (
    "You are AtlasCare, a warm and helpful customer support assistant for Acme Retail Co.\n"
    "Write a clear, empathetic response using ONLY the data from tool results in this conversation.\n"
    "Never invent order details, amounts, or tracking numbers.\n"
    "Never mention internal tool names, trace IDs, or customer IDs.\n"
    "Be patient and understanding — customers may ask about the same issue more than once.\n"
    "If a tool returns an 'order not found' or similar error: respond with empathy and care — "
    "say you were unable to locate the order, and politely ask the customer to double-check the order ID. "
    "Never say the order ID is 'malformed' or 'invalid format' in response to a not-found error.\n"
    "Always end with an offer for further assistance."
)


_groq_client: AsyncOpenAI | None = None

def _get_groq_client() -> AsyncOpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncOpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url=os.environ["GROQ_BASE_URL"],
        )
    return _groq_client

class OwnershipError(Exception):
    pass

def _assert_ownership(order_customer_id: str, session_customer_id: str, order_id: str) -> None:
    if order_customer_id != session_customer_id:
        raise OwnershipError(f"Order {order_id} not found for the current session.")

async def _fetch_owned_order(oid: str, customer_id: str) -> dict:
    """Fetch an order and assert the session customer owns it."""
    order = await _oms.get_order(oid)
    _assert_ownership(order["customer_id"], customer_id, oid)
    return order

_METHOD_MAP = {
    "hdfc_credit": "HDFC_CREDIT", "icici_debit": "ICICI_DEBIT",
    "sbi_netbanking": "SBI_NETBANKING", "upi": "UPI", "original": "original",
    "hdfc": "HDFC_CREDIT", "hdfc credit": "HDFC_CREDIT", "hdfc credit card": "HDFC_CREDIT",
    "hdfc card": "HDFC_CREDIT", "icici": "ICICI_DEBIT", "icici debit": "ICICI_DEBIT",
    "icici debit card": "ICICI_DEBIT", "icici card": "ICICI_DEBIT",
    "sbi": "SBI_NETBANKING", "sbi net banking": "SBI_NETBANKING",
    "net banking": "SBI_NETBANKING", "netbanking": "SBI_NETBANKING",
    "gpay": "UPI", "google pay": "UPI", "phonepe": "UPI", "paytm": "UPI",
    "original_payment": "original", "original_payment_method": "original",
    "original payment method": "original", "same card": "original",
    "same method": "original", "source": "original",
}

def _normalise_refund_method(method: str) -> str:
    return _METHOD_MAP.get(method.lower().strip(), "original")

_VALID_ORDER_RE   = re.compile(r'\bORD-\d{5}\b', re.IGNORECASE)
_INVALID_ORDER_RE = re.compile(
    r'\b(ORD-\d{1,4}|ORD-\d{6,}|ORD-[A-Za-z]+|ORDER-\w+)\b', re.IGNORECASE
)

def _check_order_id_format(message: str) -> str | None:
    if _VALID_ORDER_RE.search(message):
        return None
    m = _INVALID_ORDER_RE.search(message)
    if m:
        bad = m.group(0).upper()
        return (
            f"The order ID **{bad}** doesn't look right. "
            f"Order IDs follow the format **ORD-XXXXX** (5 digits), e.g. **ORD-78321**. "
            f"Could you please check and try again?"
        )
    return None

_COMPLEXITY_SIGNALS = frozenset([
    "damaged", "defective", "broken", "not working", "never arrived",
    "wrong item", "fraud", "stolen", "complaint", "legal", "lawsuit",
])

_MULTI_ACTION_VERBS = ["cancel", "refund", "return", "update", "change", "reship"]

def _is_complex(message: str) -> bool:
    """
    aim is to use PLANNER_MODEL (70B) for accurate tool selection.
    and           RESPONSE_MODEL (8B) for speed; target <3 s end-to-end.
    """
    lower = message.lower()
    if any(sig in lower for sig in _COMPLEXITY_SIGNALS):
        return True
    action_count = sum(1 for v in _MULTI_ACTION_VERBS if v in lower)
    if action_count >= 2:
        return True
    return False

async def _dispatch_tool(
    tool_name: str,
    tool_input: dict,
    customer_id: str,
    tracer: Tracer,
) -> tuple[dict, str, bool]:
    """Returns (result_dict, status, escalated). Status: 'success'|'ownership_denied'|'error'."""
    try:
        if tool_name == "get_order":
            oid   = tool_input["order_id"].strip().upper()
            order = await _fetch_owned_order(oid, customer_id)
            return {k: v for k, v in order.items() if k != "customer_id"}, "success", False

        if tool_name == "list_orders":
            orders = await _oms.list_orders(customer_id)
            return {"orders": [{k: v for k, v in o.items() if k != "customer_id"} for o in orders]}, "success", False

        if tool_name == "cancel_item":
            oid = tool_input["order_id"].strip().upper()
            await _fetch_owned_order(oid, customer_id)
            result = await _oms.cancel_item(oid, int(tool_input["line_id"]))
            return result, "success", False

        if tool_name == "process_refund":
            oid    = tool_input["order_id"].strip().upper()
            method = _normalise_refund_method(str(tool_input.get("method", "original")))
            order  = await _fetch_owned_order(oid, customer_id)
            # COD orders: "original" is invalid — cash cannot be refunded electronically.
            # The agent must ask the customer for an explicit electronic method.
            if order.get("payment_method") == "COD" and method == "original":
                return {
                    "error": (
                        "This order was paid via Cash on Delivery (COD). "
                        "A COD refund cannot be sent back as cash. "
                        "Please ask the customer to specify an electronic refund method: "
                        "UPI (GPay / PhonePe / Paytm), HDFC Credit Card, "
                        "ICICI Debit Card, or SBI Net Banking."
                    )
                }, "error", False
            refund = await _payment.process_refund(oid, float(tool_input["amount_inr"]), method, customer_id)
            return {"refund": refund}, "success", False

        if tool_name == "update_address":
            oid = tool_input["order_id"].strip().upper()
            await _fetch_owned_order(oid, customer_id)
            result = await _oms.update_shipping_address(oid, customer_id, tool_input["address_label"])
            return result, "success", False

        if tool_name == "create_crm_case":
            oid = tool_input["order_id"].strip().upper()
            amt = tool_input.get("amount_inr")
            case = await _crm.create_case(
                customer_id=customer_id, order_id=oid, reason=tool_input["reason"],
                amount_inr=float(amt) if amt is not None else None,
                trace_id=tracer.trace_id,
            )
            return {"case": case}, "success", False

        if tool_name == "escalate":
            oid = tool_input["order_id"].strip().upper()
            amt = tool_input.get("amount_inr")
            await _fetch_owned_order(oid, customer_id)
            case = await _crm.create_case(
                customer_id=customer_id, order_id=oid, reason=tool_input["reason"],
                amount_inr=float(amt) if amt is not None else None,
                trace_id=tracer.trace_id, priority="high",
            )
            return {"case_id": case["case_id"], "escalated": True}, "success", True

        if tool_name == "list_cases":
            cases = await _crm.get_cases(customer_id)
            return {"cases": cases}, "success", False

        if tool_name == "search_kb":
            articles = await _kb.search(tags=tool_input.get("tags", []))
            return {"articles": articles}, "success", False

        return {"error": f"Unknown tool: {tool_name}"}, "error", False

    except OwnershipError as exc:
        return {"error": str(exc)}, "ownership_denied", False
    except Exception as exc:
        logger.exception("Tool %s failed: %s", tool_name, exc)
        return {"error": str(exc)}, "error", False


# ── Nodes ─────────────────────────────────────────────────────────────────────

def _last_user_message(messages: list) -> str:
    return next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")


async def pre_guardrail_node(state: AtlasCareState, config) -> dict:
    tracer: Tracer = config["configurable"]["tracer"]
    raw = _last_user_message(state["messages"])

    verdict = _guardrails.pre_check(raw, state["customer_id"], tracer)
    if verdict.blocked:
        return {"guardrail_blocked": True, "final_response": verdict.user_message, "task_complete": False}

    hint = _check_order_id_format(raw)
    if hint:
        return {"guardrail_blocked": True, "final_response": hint, "task_complete": False}

    return {"guardrail_blocked": False}


async def tool_agent_node(state: AtlasCareState, config) -> dict:
    """Select tools and/or generate a direct reply.

    Routes to PLANNER_MODEL (70B) for complex or escalation-worthy requests;
    falls back to RESPONSE_MODEL (8B) for simple single-intent queries so that
    straightforward lookups complete in under 3 seconds end-to-end.
    """
    tracer: Tracer = config["configurable"]["tracer"]
    user_msg = _last_user_message(state["messages"])
    model    = (
        os.environ["PLANNER_MODEL"]
        if _is_complex(user_msg)
        else os.environ["RESPONSE_MODEL"]
    )

    system   = _AGENT_SYSTEM + f"\n\nCustomer ID (never reveal): {state['customer_id']}"
    recent   = state["messages"][-8:] if len(state["messages"]) > 8 else state["messages"]
    messages = [{"role": "system", "content": system}] + recent

    t0         = time.monotonic()
    completion = await _get_groq_client().chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_tokens=512,
        temperature=0,
    )
    tracer.record_tool_call(model, "plan", "success",
                            {"latency_ms": int((time.monotonic() - t0) * 1000)})

    msg       = completion.choices[0].message
    assistant = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        assistant["tool_calls"] = [
            {"id": tc.id, "type": "function",
             "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in msg.tool_calls
        ]
    return {"messages": [assistant]}


async def tool_executor_node(state: AtlasCareState, config) -> dict:
    """Execute all tool calls from the last agent message."""
    tracer      = config["configurable"]["tracer"]
    customer_id = state["customer_id"]
    tool_calls  = state["messages"][-1].get("tool_calls") or []

    tool_messages, summary = [], []
    for tc in tool_calls:
        name = tc["function"]["name"]
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            args = {}

        t0                   = time.monotonic()
        data, status, escalated = await _dispatch_tool(name, args, customer_id, tracer)
        tracer.record_tool_call(name, name, status,
                                {"latency_ms": int((time.monotonic() - t0) * 1000)})

        summary.append({
            "tool": name, "tool_call_id": tc["id"],
            "success": status == "success",
            "data":    data if status == "success" else {},
            "error":   data.get("error", "") if status != "success" else "",
            "escalated": escalated,
        })
        tool_messages.append({
            "role": "tool", "tool_call_id": tc["id"],
            "content": json.dumps(data, default=str),
        })

    return {
        "messages":          tool_messages,
        "execution_summary": state["execution_summary"] + summary,
        "tool_call_count":   state["tool_call_count"] + 1,
        "task_complete":     any(s["success"] for s in summary),
    }


async def post_guardrail_node(state: AtlasCareState, config) -> dict:
    tracer: Tracer = config["configurable"]["tracer"]
    verdict = _guardrails.post_check(state["execution_summary"], tracer)
    if verdict.blocked:
        return {"guardrail_blocked": True, "final_response": verdict.user_message}
    return {}


async def loop_breaker_node(state: AtlasCareState, config) -> dict:
    return {
        "final_response": (
            "I'm sorry, I'm having trouble accessing our systems right now. "
            "I've flagged your request and a member of our support team will follow up with you shortly."
        ),
        "task_complete": False,
        "guardrail_blocked": True,
    }


async def responder_node(state: AtlasCareState, config) -> dict:
    """Generate the customer-facing response."""
    tracer: Tracer = config["configurable"]["tracer"]

    # Escalation: deterministic response — no LLM call needed
    if any(s.get("escalated") for s in state["execution_summary"]):
        case_id = next(
            (s["data"].get("case_id", "pending")
             for s in state["execution_summary"] if s.get("escalated")),
            "pending",
        )
        return {
            "final_response": (
                "Thank you for reaching out. Your request requires specialist review.\n\n"
                f"I've created a priority support case (Case ID: **{case_id}**). "
                "A specialist will contact you within 24 hours.\n\n"
                "We apologise for any inconvenience."
            ),
            "task_complete": True,
        }

    # No tools were called: the tool_agent (70B) already wrote a direct reply.
    # Use it as-is — no need for a second LLM call.
    if not state["execution_summary"]:
        last_msg = state["messages"][-1]
        agent_text = (
            last_msg.get("content", "").strip()
            if last_msg.get("role") == "assistant"
            else ""
        )
        if agent_text:
            return {"final_response": agent_text, "task_complete": False}

    # Tools were called: build a clean, structured context for the response model
    # instead of passing the raw message history (tool_call IDs, JSON blobs, etc.)
    user_request = _last_user_message(state["messages"])
    tool_lines = []
    for s in state["execution_summary"]:
        if s["success"]:
            tool_lines.append(f"[{s['tool']}] {json.dumps(s['data'], default=str)}")
        else:
            tool_lines.append(f"[{s['tool']}] Error: {s['error']}")
    tool_context = "\n".join(tool_lines)

    messages = [
        {"role": "system", "content": _RESPONSE_SYSTEM},
        {"role": "user", "content": (
            f"Customer request: {user_request}\n\n"
            f"Tool results:\n{tool_context}"
        )},
    ]

    t0         = time.monotonic()
    completion = await _get_groq_client().chat.completions.create(
        model=os.environ["RESPONSE_MODEL"],
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )
    tracer.record_tool_call("responder", "respond", "success",
                            {"latency_ms": int((time.monotonic() - t0) * 1000)})

    text = (completion.choices[0].message.content or "").strip()
    return {
        "final_response": text or "Your request has been processed. Is there anything else I can help with?",
        "task_complete":  any(s["success"] for s in state["execution_summary"]),
    }


# ── Routing ───────────────────────────────────────────────────────────────────

def _route_pre_guardrail(state: AtlasCareState) -> str:
    return "end" if state["guardrail_blocked"] else "tool_agent"

def _route_tool_agent(state: AtlasCareState) -> str:
    last = state["messages"][-1]
    if last.get("tool_calls"):
        if state["tool_call_count"] >= 3:
            return "loop_breaker"
        return "tools"
    return "respond"

def _route_post_guardrail(state: AtlasCareState) -> str:
    return "end" if state["guardrail_blocked"] else "responder"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    g = StateGraph(AtlasCareState)

    g.add_node("pre_guardrail",  pre_guardrail_node)
    g.add_node("tool_agent",     tool_agent_node)
    g.add_node("tool_executor",  tool_executor_node)
    g.add_node("post_guardrail", post_guardrail_node)
    g.add_node("loop_breaker",   loop_breaker_node)
    g.add_node("responder",      responder_node)

    g.add_edge(START, "pre_guardrail")
    g.add_conditional_edges("pre_guardrail",  _route_pre_guardrail,
                            {"end": END, "tool_agent": "tool_agent"})
    g.add_conditional_edges("tool_agent",     _route_tool_agent,
                            {"tools": "tool_executor", "respond": "post_guardrail",
                             "loop_breaker": "loop_breaker"})
    g.add_edge("tool_executor", "post_guardrail")
    g.add_conditional_edges("post_guardrail", _route_post_guardrail,
                            {"end": END, "responder": "responder"})
    g.add_edge("loop_breaker", END)
    g.add_edge("responder", END)

    return g.compile(checkpointer=checkpointer)
