"""
AtlasCare — Agentic AI Customer Support Platform
Entry point: FastAPI application

Exposes:
  POST /query  — main agent interaction endpoint
  GET  /health — liveness probe
"""

import time
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models.request_models import QueryRequest
from models.response_models import QueryResponse, TraceModel
from agent.orchestrator import Orchestrator
from observability.logger import configure_logging
from observability.tracer import Tracer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan — warm up / tear down shared resources
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: validate required env vars, initialise the orchestrator singleton.
    Shutdown: flush any in-flight observability buffers.
    """
    _assert_env_vars()
    logger.info("AtlasCare starting up — environment validated.")

    # Attach a single orchestrator instance to app state so it can be
    # reused across requests without re-building the tool graph every call.
    app.state.orchestrator = Orchestrator()
    logger.info("Orchestrator initialised.")

    yield

    logger.info("AtlasCare shutting down.")


def _assert_env_vars() -> None:
    """Fail fast if mandatory environment variables are absent."""
    required = [
        "GEMINI_API_KEY",        # Gemini 2.5 Flash via OpenAI-compatible endpoint
        "GEMINI_BASE_URL",       # e.g. https://generativelanguage.googleapis.com/v1beta/openai
        "GEMINI_MODEL",          # e.g. gemini-2.5-flash
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"AtlasCare: missing required environment variable(s): {missing}. "
            "Check your .env file or deployment config."
        )


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AtlasCare",
    description="Agentic AI customer support platform for Acme Retail Co.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Return a clean 422 instead of FastAPI's default verbose payload."""
    logger.warning("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Catch-all — prevent stack traces leaking to callers.
    All unhandled exceptions are logged server-side; the client
    receives a safe generic message.
    """
    logger.exception("Unhandled exception for request %s", request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again later."},
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    tags=["ops"],
)
async def health():
    """
    Kubernetes / ELB liveness check.
    Returns HTTP 200 as long as the process is alive.
    """
    return {"status": "ok"}


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a customer query to the AtlasCare agent",
    tags=["agent"],
)
async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    """
    Main agent endpoint.

    Contract
    --------
    Request  : { "message": str, "session_id": str }
    Response : { "response": str, "trace": { trace_id, session_id, latency_ms, tool_calls } }

    The orchestrator handles intent planning, tool execution, guardrails,
    escalation logic, and response assembly.  This layer is intentionally
    thin — it owns only HTTP concerns (timing, logging, error surfacing).
    """

    # use monotonic as it is unidirectional internal clock immune to maunal changes in system clock
    wall_start = time.monotonic()

    tracer = Tracer(session_id=request.session_id)
    logger.info(
        "Received query | session=%s | trace=%s | message_preview=%.80r",
        request.session_id,
        tracer.trace_id,
        request.message,
    )
    
    # handoff from APIs to agent
    orchestrator: Orchestrator = http_request.app.state.orchestrator
    result = await orchestrator.handle(
        message=request.message,
        session_id=request.session_id,
        tracer=tracer,
    )

    latency_ms = int((time.monotonic() - wall_start) * 1000)
    tracer.set_latency(latency_ms)

    logger.info(
        "Query completed | session=%s | trace=%s | latency_ms=%d | tools_called=%d",
        request.session_id,
        tracer.trace_id,
        latency_ms,
        len(tracer.tool_calls),
    )

    return QueryResponse(
        response=result.response_text,
        trace=TraceModel(
            trace_id=tracer.trace_id,
            session_id=request.session_id,
            latency_ms=latency_ms,
            tool_calls=tracer.tool_calls,
        ),
    )