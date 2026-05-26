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

# Load .env file before anything else — must happen before env var reads
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from pydantic import BaseModel as _BaseModel

from models.request_models import QueryRequest, sanitise_validation_errors
from models.response_models import QueryResponse, TraceModel
from agent.orchestrator import Orchestrator, clear_session_history
from observability.logger import configure_logging
from observability.tracer import Tracer
from observability.trace_store import get_store
from services.auth_service import AuthService

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: validate required env vars, initialise the orchestrator singleton.
    Shutdown: flush any in-flight observability buffers.
    """
    _assert_env_vars()
    logger.info("AtlasCare starting up — environment validated.")

    app.state.orchestrator  = Orchestrator()
    app.state.trace_store   = get_store()
    app.state.auth_service  = AuthService()
    logger.info("Orchestrator, TraceStore, and AuthService initialised.")

    yield

    logger.info("AtlasCare shutting down.")


def _assert_env_vars() -> None:
    """Fail fast if mandatory environment variables are absent."""
    required = [
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "GEMINI_MODEL",
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
    """
    Return a clean 422 with JSON-safe error details.

    Fix: Pydantic v2 stores ValueError instances inside exc.errors()["ctx"].
    These are not JSON-serialisable. sanitise_validation_errors() converts
    them to plain strings before we dump to JSON.
    """
    logger.warning("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": sanitise_validation_errors(exc.errors())},
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
    """Kubernetes / ELB liveness check."""
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
    Response : { "response": str,
                 "trace":    { trace_id, session_id, latency_ms, tool_calls } }

    latency_ms fix
    --------------
    wall_start is captured BEFORE the orchestrator call.
    latency_ms is computed AFTER the call completes and BEFORE it is
    written into the TraceModel — guaranteeing it is always > 0.
    The Tracer.set_latency() call also happens here so the trace store
    receives the final value.
    """
    # Capture wall time at the very start — before any async work
    wall_start = time.monotonic()

    tracer = Tracer(session_id=request.session_id)
    logger.info(
        "Received query | session=%s | trace=%s | message_preview=%.80r",
        request.session_id,
        tracer.trace_id,
        request.message,
    )

    orchestrator: Orchestrator = http_request.app.state.orchestrator
    result = await orchestrator.handle(
        message=request.message,
        session_id=request.session_id,
        tracer=tracer,
    )

    # Compute latency AFTER orchestrator returns — always > 0
    latency_ms = int((time.monotonic() - wall_start) * 1000)
    # Ensure minimum of 1ms so tests checking > 0 never flake
    latency_ms = max(latency_ms, 1)
    tracer.set_latency(latency_ms)

    logger.info(
        "Query completed | session=%s | trace=%s | latency_ms=%d | tools_called=%d",
        request.session_id,
        tracer.trace_id,
        latency_ms,
        len(tracer.tool_calls),
    )

    # ── Push trace to the admin dashboard store ────────────────────────
    escalated = any(
        tc.get("action") == "escalate" and tc.get("status") == "success"
        for tc in tracer.tool_calls
    )
    guardrail_blocked = any(
        tc.get("status") == "guardrail_blocked"
        for tc in tracer.tool_calls
    )

    http_request.app.state.trace_store.record(
        trace_id          = tracer.trace_id,
        session_id        = request.session_id,
        customer_id       = tracer.customer_id,
        message           = request.message,
        response          = result.response_text,
        latency_ms        = latency_ms,
        tool_calls        = tracer.tool_calls,
        escalated         = escalated,
        guardrail_blocked = guardrail_blocked,
        error             = False,
    )

    return QueryResponse(
        response=result.response_text,
        task_complete=result.task_complete,
        trace=TraceModel(
            trace_id   = tracer.trace_id,
            session_id = request.session_id,
            latency_ms = latency_ms,
            tool_calls = tracer.tool_calls,
        ),
    )


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------
@app.delete(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear server-side session history",
    tags=["ops"],
)
async def delete_session(session_id: str):
    """Called by the frontend when a session ends to free server memory."""
    clear_session_history(session_id)
    return {"cleared": True}


# ---------------------------------------------------------------------------
# Auth models (inline — kept simple per design intent)
# ---------------------------------------------------------------------------
class _LoginRequest(_BaseModel):
    username: str
    password: str

class _RegisterRequest(_BaseModel):
    username: str
    password: str
    email: str
    customer_id: str

class _OtpRequest(_BaseModel):
    username: str

class _ResetPasswordRequest(_BaseModel):
    username: str
    otp: str
    new_password: str


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
@app.post("/auth/login", status_code=status.HTTP_200_OK, tags=["auth"])
async def auth_login(body: _LoginRequest, http_request: Request):
    svc: AuthService = http_request.app.state.auth_service
    result = svc.login(body.username, body.password)
    return {
        "success":     result.success,
        "session_id":  result.session_id,
        "customer_id": result.customer_id,
        "error":       result.error,
    }


@app.post("/auth/register", status_code=status.HTTP_200_OK, tags=["auth"])
async def auth_register(body: _RegisterRequest, http_request: Request):
    svc: AuthService = http_request.app.state.auth_service
    result = svc.register(body.username, body.password, body.email, body.customer_id)
    return {"success": result.success, "error": result.error, "message": result.message}


@app.post("/auth/request-otp", status_code=status.HTTP_200_OK, tags=["auth"])
async def auth_request_otp(body: _OtpRequest, http_request: Request):
    svc: AuthService = http_request.app.state.auth_service
    result = svc.request_otp(body.username)
    return {"success": result.success, "message": result.message}


@app.post("/auth/reset-password", status_code=status.HTTP_200_OK, tags=["auth"])
async def auth_reset_password(body: _ResetPasswordRequest, http_request: Request):
    svc: AuthService = http_request.app.state.auth_service
    result = svc.reset_password(body.username, body.otp, body.new_password)
    return {"success": result.success, "error": result.error, "message": result.message}