"""
agent/planner.py
================
LLM-assisted intent extraction and action plan generation.

Responsibility
--------------
  1. Send the customer message to Gemini 2.5 Flash with a structured
     system prompt that constrains output to a JSON action plan.
  2. Parse and validate the JSON response deterministically.
  3. Return a typed ActionPlan consumed by the Executor.

Design principles
-----------------
- The LLM is used ONLY to understand intent and structure steps.
  It NEVER decides policy (thresholds, ownership, auth).
- Output is always validated against a strict schema before use.
  Malformed LLM output raises PlannerError — never silently continues.
- The planner is stateless; all context comes from the message +
  customer_id passed per call.
- Prompt is versioned via SYSTEM_PROMPT_VERSION for auditability.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openai import AsyncOpenAI  # Gemini via OpenAI-compatible endpoint
from observability.tracer import Tracer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_VERSION = "v1.0"

# ---------------------------------------------------------------------------
# Domain enums — intent taxonomy
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    ORDER_TRACKING       = "order_tracking"
    PARTIAL_CANCELLATION = "partial_cancellation"
    REFUND_REQUEST       = "refund_request"
    ADDRESS_UPDATE       = "address_update"
    COMPOUND             = "compound"          # multi-step combination
    POLICY_QUERY         = "policy_query"
    ESCALATION           = "escalation"
    UNKNOWN              = "unknown"


class ActionType(str, Enum):
    GET_ORDER            = "get_order"
    CANCEL_ITEM          = "cancel_item"
    PROCESS_REFUND       = "process_refund"
    UPDATE_ADDRESS       = "update_address"
    CREATE_CRM_CASE      = "create_crm_case"
    SEARCH_KB            = "search_kb"
    ESCALATE             = "escalate"


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------
@dataclass
class ActionStep:
    """
    A single deterministic step the Executor will carry out.

    Parameters
    ----------
    action      : what tool action to invoke
    params      : free-form dict the Executor passes to the tool
    depends_on  : step indices (0-based) that must succeed before this runs
    """
    action: ActionType
    params: dict[str, Any]
    depends_on: list[int] = field(default_factory=list)


@dataclass
class ActionPlan:
    """
    Structured output of the Planner consumed by the Executor.

    Attributes
    ----------
    intent      : top-level classified intent
    steps       : ordered list of ActionSteps
    raw_llm     : original LLM JSON string (stored for auditability)
    """
    intent: Intent
    steps: list[ActionStep]
    raw_llm: str = ""


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class PlannerError(Exception):
    """Raised when the planner cannot produce a valid plan."""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = f"""
You are AtlasCare, an AI assistant for Acme Retail customer support.
System prompt version: {SYSTEM_PROMPT_VERSION}

Your ONLY job in this step is to analyse the customer message and return
a structured JSON action plan. You must NOT answer the customer directly.

---
AVAILABLE ACTIONS
-----------------
- get_order          : params: {{ "order_id": str }}
- cancel_item        : params: {{ "order_id": str, "line_id": int }}
- process_refund     : params: {{ "order_id": str, "amount_inr": float, "method": str }}
- update_address     : params: {{ "order_id": str, "address_label": str }}
- create_crm_case    : params: {{ "order_id": str, "reason": str, "amount_inr": float | null }}
- search_kb          : params: {{ "tags": [str] }}
- escalate           : params: {{ "order_id": str, "reason": str, "amount_inr": float | null }}

---
INTENT TAXONOMY
---------------
order_tracking | partial_cancellation | refund_request |
address_update | compound | policy_query | escalation | unknown

---
OUTPUT FORMAT (strict JSON, no markdown, no extra keys)
--------------------------------------------------------
{{
  "intent": "<intent>",
  "steps": [
    {{
      "action": "<action_type>",
      "params": {{ ... }},
      "depends_on": [<step indices>]
    }}
  ]
}}

---
RULES
-----
1. Return ONLY valid JSON. No explanations, no markdown fences.
2. For compound requests, produce multiple steps in dependency order.
3. If a refund amount is mentioned, include amount_inr in params.
4. If the customer references "my office address", use address_label = "office".
5. If intent is unclear, set intent = "unknown" and steps = [].
6. Never fabricate order IDs, amounts, or addresses not stated in the message.
7. Do NOT decide whether a refund is allowed — only plan the step; guardrails enforce policy.
""".strip()


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
class Planner:
    """
    Calls Gemini 2.5 Flash to convert a raw customer message into a
    typed ActionPlan.

    Thread / async safe — no mutable instance state after __init__.
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=os.environ["GEMINI_BASE_URL"],
        )
        self._model = os.environ["GEMINI_MODEL"]
        logger.debug("Planner initialised with model=%s", self._model)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    async def plan(
        self,
        message: str,
        customer_id: str,
        tracer: Tracer,
    ) -> ActionPlan:
        """
        Produce an ActionPlan for the given customer message.

        Raises
        ------
        PlannerError  if the LLM returns malformed JSON or an unknown
                      action type — callers should catch this.
        """
        raw = await self._call_llm(message, tracer)
        plan = self._parse_and_validate(raw)

        tracer.record_tool_call(
            tool="planner",
            action="plan",
            status="success",
            meta={
                "intent": plan.intent,
                "steps": len(plan.steps),
                "prompt_version": SYSTEM_PROMPT_VERSION,
            },
        )

        return plan

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    async def _call_llm(self, message: str, tracer: Tracer) -> str:
        """
        Send message to Gemini and return raw response text.

        Uses temperature=0 for maximum determinism in planning output.
        """
        import time
        t0 = time.monotonic()

        try:
            completion = await self._client.chat.completions.create(
                model=self._model,
                temperature=0,
                max_tokens=1024,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": message},
                ],
            )
        except Exception as exc:
            tracer.record_tool_call(
                tool="planner",
                action="llm_call",
                status="error",
                meta={"error": str(exc)},
            )
            raise PlannerError(f"LLM call failed: {exc}") from exc

        latency_ms = int((time.monotonic() - t0) * 1000)
        raw = completion.choices[0].message.content or ""

        logger.debug(
            "LLM response | trace=%s | latency_ms=%d | raw=%.200r",
            tracer.trace_id,
            latency_ms,
            raw,
        )
        return raw

    def _parse_and_validate(self, raw: str) -> ActionPlan:
        """
        Parse raw LLM JSON into a typed ActionPlan.

        Raises PlannerError on any structural or type violation.
        This is intentionally strict — bad plans must never reach
        the Executor.
        """
        # Strip accidental markdown fences the model may emit
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise PlannerError(
                f"LLM returned non-JSON output: {exc}\nRaw: {raw[:300]}"
            ) from exc

        # Validate top-level shape
        if not isinstance(data, dict):
            raise PlannerError("LLM plan is not a JSON object.")
        if "intent" not in data or "steps" not in data:
            raise PlannerError("LLM plan missing 'intent' or 'steps' keys.")

        # Validate intent
        try:
            intent = Intent(data["intent"])
        except ValueError:
            logger.warning(
                "Unknown intent '%s' from LLM — defaulting to UNKNOWN",
                data["intent"],
            )
            intent = Intent.UNKNOWN

        # Validate steps
        steps: list[ActionStep] = []
        for i, raw_step in enumerate(data.get("steps", [])):
            if not isinstance(raw_step, dict):
                raise PlannerError(f"Step {i} is not a dict.")
            if "action" not in raw_step:
                raise PlannerError(f"Step {i} missing 'action' key.")
            try:
                action = ActionType(raw_step["action"])
            except ValueError:
                raise PlannerError(
                    f"Step {i} has unknown action '{raw_step['action']}'."
                )
            params = raw_step.get("params", {})
            if not isinstance(params, dict):
                raise PlannerError(f"Step {i} 'params' is not a dict.")
            depends_on = raw_step.get("depends_on", [])
            if not isinstance(depends_on, list):
                depends_on = []

            steps.append(ActionStep(
                action=action,
                params=params,
                depends_on=[int(d) for d in depends_on],
            ))

        return ActionPlan(intent=intent, steps=steps, raw_llm=raw)