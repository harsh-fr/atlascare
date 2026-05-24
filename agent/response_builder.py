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
3. Be concise but complete. Bullet points are fine for multi-step outcomes.
4. Use INR amounts exactly as provided — do not round or reformat.
5. If any step failed, acknowledge it honestly and suggest the customer
   try again or contact support.
6. Never mention internal system names (OMS, CRM, PaymentTool, trace_id,
   step indices, etc.) in the customer-facing response.
7. Always end with a helpful closing line offering further assistance.
8. Do not repeat the customer's message back to them.
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

        For escalation cases: returns a deterministic template —
        the LLM is NOT called, guaranteeing audit consistency.

        For all other cases: calls Gemini with verified tool output
        data only, then validates the response is non-empty.

        Falls back to ExecutionResult.fallback_summary() on any error.
        """

        # ----------------------------------------------------------
        # Escalation — fully deterministic, no LLM
        # ----------------------------------------------------------
        if execution_result.escalated:
            return self._escalation_response(execution_result)

        # ----------------------------------------------------------
        # All other intents — LLM-assisted phrasing
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
        """
        Returns a consistent, audit-safe escalation message.
        The LLM is never used here — wording must be stable for
        compliance and QA review.
        """
        case_id = result.escalation_case_id or "pending"
        return (
            "Thank you for getting in touch. Your request requires review "
            "by one of our specialist agents.\n\n"
            f"I've created a priority support case for you (Case ID: **{case_id}**). "
            "A specialist will contact you within 24 hours to resolve this.\n\n"
            "We apologise for any inconvenience and appreciate your patience."
        )

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
    # LLM call
    # ------------------------------------------------------------------
    async def _call_llm(self, user_prompt: str, tracer: Tracer) -> str:
        import time
        t0 = time.monotonic()

        completion = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.2,
            max_tokens=512,
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