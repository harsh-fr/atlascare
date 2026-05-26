"""
agent/response_builder.py
=========================
LLM-assisted final response assembly.

Responsibility
--------------
  1. Take the execution results (tool outputs, step outcomes) and
     assemble a grounded, accurate, customer-facing response.
  2. Use Gemini ONLY for natural language phrasing — never for
     deciding what happened or fabricating missing data.
  3. If the LLM call fails, fall back to a deterministic template
     so the customer always receives a reply.

Design principles
-----------------
- Anti-hallucination by construction: the LLM receives only
  verified tool output data in its context. It cannot invent
  order details, tracking numbers, amounts, or case IDs because
  it only sees what the tools actually returned.
- The prompt explicitly instructs the model to use ONLY the
  provided data and to say "I don't have that information" rather
  than guess.
- Escalation responses are fully deterministic — the LLM is NOT
  used for escalation phrasing to guarantee audit consistency.
- Temperature is set low (0.2) to keep phrasing consistent while
  allowing natural variation.
"""

import logging
import os
import json
from typing import Any

from openai import AsyncOpenAI
from agent.planner import ActionPlan, Intent, ActionType
from agent.executor import ExecutionResult, StepResult
from observability.tracer import Tracer

logger = logging.getLogger(__name__)

RESPONSE_PROMPT_VERSION = "v1.0"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = f"""
You are AtlasCare, a helpful and professional customer support assistant
for Acme Retail Co.
Response prompt version: {RESPONSE_PROMPT_VERSION}

Your job is to write a clear, warm, and accurate response to the customer
based ONLY on the verified data provided to you below.

STRICT RULES
------------
1. Use ONLY the data in the <verified_data> block. Never invent, guess,
   or assume any details not explicitly present in that block.
2. If a piece of information is missing from verified_data, say
   "I don't have that information right now" — do not fabricate it.
3. Be concise but complete. Never cut off mid-sentence. Always finish
   your response fully before stopping.
4. Use INR amounts exactly as provided — do not round or reformat.
5. ORDER NOT FOUND: If a step failed because the order was not found,
   clearly tell the customer: "I could not find order <ID> on your account.
   Please check the order ID and try again." Do not say "system error".
6. INVALID ORDER ID: If intent is "unknown" and no steps were planned,
   tell the customer their order ID format looks incorrect and give
   an example: "Order IDs follow the format ORD-XXXXX, for example
   ORD-78321. Could you please check and try again?"
7. If any step failed due to a refund method issue, list the valid
   methods clearly: HDFC_CREDIT, ICICI_DEBIT, SBI_NETBANKING, UPI, original.
8. If any step failed, acknowledge it honestly and suggest next steps.
9. Never mention internal system names (OMS, CRM, PaymentTool, trace_id,
   step indices, etc.) in the customer-facing response.
10. Always end with a helpful closing line offering further assistance.
11. Do not repeat the customer's message back to them.
12. Write complete sentences. Never truncate your response.
""".strip()


# ---------------------------------------------------------------------------
# ResponseBuilder
# ---------------------------------------------------------------------------
class ResponseBuilder:
    """
    Builds the final customer-facing response string.

    Stateless — all context is passed per call.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=os.environ["GEMINI_BASE_URL"],
        )
        self._model = os.environ["GEMINI_MODEL"]
        logger.debug("ResponseBuilder initialised with model=%s", self._model)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def build(
        self,
        message: str,
        plan: ActionPlan,
        execution_result: ExecutionResult,
        customer_id: str,
        tracer: Tracer,
    ) -> str:
        """
        Produce the final customer-facing response string.

        Fast-path (no LLM call, deterministic):
          - Escalation cases       → fixed template
          - Order tracking success → built from tool data
          - Order not found        → clear not-found message
          - All steps failed       → clear error message

        LLM-assisted (Gemini called):
          - Compound requests, refunds, address updates, KB queries
        """

        # ----------------------------------------------------------
        # Fast-path 1 — Escalation (deterministic, no LLM)
        # ----------------------------------------------------------
        if execution_result.escalated:
            return self._escalation_response(execution_result)

        # ----------------------------------------------------------
        # Fast-path 2 — Order tracking success (no LLM needed)
        # Saves ~2-3 seconds on the most common query type.
        # ----------------------------------------------------------
        if (
            plan.intent == Intent.ORDER_TRACKING
            and execution_result.overall_success()
        ):
            order_data = execution_result.get_step_data(ActionType.GET_ORDER)
            if order_data and "order" in order_data:
                return self._order_tracking_response(order_data["order"])

        # ----------------------------------------------------------
        # Fast-path 3 — Order not found (deterministic, no LLM)
        # ----------------------------------------------------------
        if plan.intent == Intent.ORDER_TRACKING and not execution_result.any_success():
            failed = execution_result.failed_steps()
            if failed:
                # Extract order_id from the failed step params
                step_index = failed[0].step_index
                order_id = "your order"
                if step_index < len(plan.steps):
                    order_id = plan.steps[step_index].params.get("order_id", "your order")
                return (
                    f"I could not find order **{order_id}** on your account. "
                    "Please check the order ID and try again. "
                    "Order IDs follow the format **ORD-XXXXX**, for example **ORD-78321**. "
                    "You can find your order ID in your confirmation email or order history."
                )

        # ----------------------------------------------------------
        # LLM-assisted — compound, refund, address, KB, unknown
        # ----------------------------------------------------------
        verified_data = self._build_verified_data(plan, execution_result)
        user_prompt   = self._build_user_prompt(
            message=message,
            intent=plan.intent,
            verified_data=verified_data,
            execution_result=execution_result,
        )

        try:
            response_text = await self._call_llm(user_prompt, tracer)
        except Exception as exc:
            logger.exception(
                "ResponseBuilder LLM call failed | trace=%s | error=%s",
                tracer.trace_id,
                exc,
            )
            return execution_result.fallback_summary()

        if not response_text or not response_text.strip():
            logger.warning(
                "ResponseBuilder received empty LLM response | trace=%s",
                tracer.trace_id,
            )
            return execution_result.fallback_summary()

        return response_text.strip()

    # ------------------------------------------------------------------
    # Deterministic escalation response
    # ------------------------------------------------------------------
    def _escalation_response(self, result: ExecutionResult) -> str:
        case_id = result.escalation_case_id or "pending"
        return (
            "Thank you for getting in touch. Your request requires review "
            "by one of our specialist agents.\n\n"
            f"I've created a priority support case for you (Case ID: **{case_id}**). "
            "A specialist will contact you within 24 hours to resolve this.\n\n"
            "We apologise for any inconvenience and appreciate your patience."
        )

    # ------------------------------------------------------------------
    # Deterministic order tracking response (fast-path, no LLM)
    # ------------------------------------------------------------------
    def _order_tracking_response(self, order: dict) -> str:
        """
        Build a complete order status response from verified tool data.
        Zero LLM calls — deterministic, instant, grounded.
        """
        order_id   = order.get("order_id", "your order")
        status     = order.get("status", "unknown").capitalize()
        tracking   = order.get("tracking_number")
        estimated  = order.get("estimated_delivery", "")
        items      = order.get("items", [])
        total      = order.get("total_amount", 0)
        payment    = order.get("payment_method", "")

        # Status-specific message
        _STATUS_MESSAGES = {
            "placed":     "Your order has been placed and is awaiting processing.",
            "processing": "Your order is currently being processed and will be shipped soon.",
            "shipped":    "Great news — your order is on its way!",
            "delivered":  "Your order has been delivered.",
            "cancelled":  "Your order has been cancelled.",
        }
        status_msg = _STATUS_MESSAGES.get(
            order.get("status", "").lower(),
            f"Your order status is: {status}."
        )

        lines = [f"Here's the update on order **{order_id}**:\n"]
        lines.append(f"**Status:** {status_msg}")

        if tracking:
            lines.append(f"**Tracking Number:** {tracking}")

        if estimated and order.get("status") not in ("delivered", "cancelled"):
            lines.append(f"**Estimated Delivery:** {estimated}")

        # Active items summary
        active_items = [i for i in items if i.get("status") == "active"]
        if active_items:
            lines.append(f"**Items ({len(active_items)}):**")
            for item in active_items:
                lines.append(
                    f"  - {item.get('name', 'Item')} "
                    f"(x{item.get('quantity', 1)}) — "
                    f"₹{item.get('unit_price', 0):,.2f}"
                )

        lines.append(f"**Order Total:** ₹{total:,.2f}")

        lines.append(
            "\nIs there anything else I can help you with?"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Verified data assembly — grounding the LLM
    # ------------------------------------------------------------------
    def _build_verified_data(
        self,
        plan: ActionPlan,
        result: ExecutionResult,
    ) -> dict[str, Any]:
        """
        Build a clean dict of verified tool outputs to pass to the LLM.

        Only successful step data is included. Failed steps are
        represented as error entries so the LLM can acknowledge them
        without fabricating outcomes.
        """
        verified: dict[str, Any] = {
            "intent": plan.intent.value,
            "steps":  [],
        }

        for sr in result.step_results:
            entry: dict[str, Any] = {
                "action":  sr.action.value,
                "success": sr.success,
            }
            if sr.success:
                entry["data"] = _sanitise_for_llm(sr.data)
            else:
                entry["error"] = sr.error or "Unknown error."
            verified["steps"].append(entry)

        return verified

    # ------------------------------------------------------------------
    # User prompt construction
    # ------------------------------------------------------------------
    def _build_user_prompt(
        self,
        message: str,
        intent: Intent,
        verified_data: dict[str, Any],
        execution_result: ExecutionResult,
    ) -> str:
        """
        Construct the user-turn prompt containing verified data only.
        """
        failed = execution_result.failed_steps()
        all_ok = execution_result.overall_success()

        status_hint = (
            "All requested actions completed successfully."
            if all_ok
            else f"{len(failed)} action(s) could not be completed."
        )

        return (
            f"Customer message: {message}\n\n"
            f"Intent detected: {intent.value}\n"
            f"Execution status: {status_hint}\n\n"
            "<verified_data>\n"
            f"{json.dumps(verified_data, indent=2, default=str)}\n"
            "</verified_data>\n\n"
            "Please write a helpful, accurate response to the customer "
            "using ONLY the data above."
        )

    # ------------------------------------------------------------------
    # Conversational response (greetings, chitchat)
    # ------------------------------------------------------------------
    _CONVERSE_SYSTEM = (
        "You are AtlasCare, a warm and friendly customer support assistant for "
        "Acme Retail Co. When a customer greets you, welcome them naturally and "
        "ask how you can help today. Keep responses concise (2-3 sentences). "
        "Do not ask for an Order ID unprompted — just engage conversationally."
    )

    async def converse(
        self,
        message: str,
        history: list[dict],
        tracer: Tracer,
    ) -> str:
        """Direct LLM call for conversational/greeting messages."""
        messages: list[dict] = [{"role": "system", "content": self._CONVERSE_SYSTEM}]
        for turn in history[-4:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")[:200]
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": message})

        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.7,
            max_tokens=200,
            messages=messages,
        )
        return (completion.choices[0].message.content or "").strip() or (
            "Hello! Welcome to AtlasCare. How can I help you today?"
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------
    async def _call_llm(self, user_prompt: str, tracer: Tracer) -> str:
        import time
        t0 = time.monotonic()

        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        latency_ms = int((time.monotonic() - t0) * 1000)
        raw = completion.choices[0].message.content or ""

        tracer.record_tool_call(
            tool="response_builder",
            action="llm_phrase",
            status="success",
            meta={
                "latency_ms": latency_ms,
                "prompt_version": RESPONSE_PROMPT_VERSION,
                "output_chars": len(raw),
            },
        )

        logger.debug(
            "ResponseBuilder LLM done | trace=%s | latency_ms=%d | chars=%d",
            tracer.trace_id,
            latency_ms,
            len(raw),
        )
        return raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sanitise_for_llm(data: dict[str, Any]) -> dict[str, Any]:
    """
    Strip internal/sensitive fields from tool output before
    passing to the LLM context.

    Removes: customer_id, raw JSON blobs, internal IDs that should
    not appear in customer-facing text.
    """
    _STRIP_KEYS = {"customer_id", "raw_llm", "trace_id", "password"}

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {
                k: _clean(v)
                for k, v in obj.items()
                if k not in _STRIP_KEYS
            }
        if isinstance(obj, list):
            return [_clean(i) for i in obj]
        return obj

    return _clean(data)