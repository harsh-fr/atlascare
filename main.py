import time
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel as _BaseModel

from langgraph.checkpoint.memory import MemorySaver

from models.request_models import QueryRequest, sanitise_validation_errors
from models.response_models import QueryResponse, TraceModel
from agent.graph import build_graph, _crm as _graph_crm, _kb as _graph_kb
from observability.logger import configure_logging
from observability.tracer import Tracer
from observability.trace_store import get_store
from services.auth_service import AuthService
from utils.session_store import SessionStore
from tools.crm_tool import CrmTool
from tools.kb_tool import KbTool

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _assert_env_vars()
    logger.info("AtlasCare starting up — environment validated.")

    checkpointer            = MemorySaver()
    app.state.graph         = build_graph(checkpointer)
    app.state.checkpointer  = checkpointer
    app.state.trace_store   = get_store()
    app.state.auth_service  = AuthService()
    app.state.session_store = SessionStore()
    app.state.crm_tool      = _graph_crm   # shared with agent — no duplicate instance
    app.state.kb_tool       = _graph_kb
    logger.info("Graph, TraceStore, AuthService, and SessionStore initialised.")

    yield

    logger.info("AtlasCare shutting down.")


def _assert_env_vars() -> None:
    required = ["GROQ_API_KEY", "GROQ_BASE_URL", "PLANNER_MODEL", "RESPONSE_MODEL"]
    missing  = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"AtlasCare: missing required environment variable(s): {missing}. "
            "Check your .env file or deployment config."
        )


app = FastAPI(
    title="AtlasCare",
    description="Agentic AI customer support platform for Acme Retail Co.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    logger.warning("Request validation failed: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": sanitise_validation_errors(exc.errors())},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception for request %s", request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred. Please try again later."},
    )

# health route
@app.get("/health", status_code=status.HTTP_200_OK, summary="Liveness probe", tags=["ops"])
async def health():
    return {"status": "ok"}


@app.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a customer query to the AtlasCare agent",
    tags=["agent"],
)
async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    wall_start = time.monotonic()
    tracer = Tracer(session_id=request.session_id)
    logger.info(
        "Received query | session=%s | trace=%s | message_preview=%.80r",
        request.session_id, tracer.trace_id, request.message,
    )

    session_store: SessionStore = http_request.app.state.session_store
    customer_id = session_store.resolve(request.session_id)

    if customer_id is None:
        logger.warning(
            "Unresolvable session | trace=%s | session=%s",
            tracer.trace_id, request.session_id,
        )
        response_text = (
            "I'm unable to verify your session. "
            "Please log in again and retry."
        )
        task_complete = False
    else:
        tracer.set_customer_id(customer_id)
        initial_state = {
            "messages":          [{"role": "user", "content": request.message}],
            "session_id":        request.session_id,
            "customer_id":       customer_id,
            "guardrail_blocked": False,
            "execution_summary": [],
            "tool_call_count":   0,
            "final_response":    "",
            "task_complete":     False,
        }
        config = {"configurable": {"thread_id": request.session_id, "tracer": tracer}}
        try:
            final_state = await http_request.app.state.graph.ainvoke(
                initial_state, config=config,
            )
            response_text = final_state["final_response"]
            task_complete = final_state.get("task_complete", False)
        except Exception as exc:
            logger.exception(
                "Agent execution failed | trace=%s | error=%s",
                tracer.trace_id, exc,
            )
            response_text = (
                "I encountered an issue processing your request. "
                "Please try again."
            )
            task_complete = False

    latency_ms = max(int((time.monotonic() - wall_start) * 1000), 1)
    tracer.set_latency(latency_ms)
    logger.info(
        "Query completed | session=%s | trace=%s | latency_ms=%d | tools_called=%d",
        request.session_id, tracer.trace_id, latency_ms, len(tracer.tool_calls),
    )

    escalated = any(
        tc.get("action") == "escalate" and tc.get("status") == "success"
        for tc in tracer.tool_calls
    )
    guardrail_blocked = any(
        tc.get("status") == "guardrail_blocked" for tc in tracer.tool_calls
    )

    http_request.app.state.trace_store.record(
        trace_id=tracer.trace_id,
        session_id=request.session_id,
        customer_id=tracer.customer_id,
        message=request.message,
        response=response_text,
        latency_ms=latency_ms,
        tool_calls=tracer.tool_calls,
        escalated=escalated,
        guardrail_blocked=guardrail_blocked,
        error=False,
    )

    return QueryResponse(
        response=response_text,
        task_complete=task_complete,
        trace=TraceModel(
            trace_id=tracer.trace_id,
            session_id=request.session_id,
            latency_ms=latency_ms,
            tool_calls=tracer.tool_calls,
        ),
    )


# ---------------------------------------------------------------------------
# Admin read endpoints
# ---------------------------------------------------------------------------
@app.get("/admin/traces", status_code=status.HTTP_200_OK, tags=["admin"])
async def admin_traces(http_request: Request):
    return http_request.app.state.trace_store.get_all()


@app.get("/admin/kpis", status_code=status.HTTP_200_OK, tags=["admin"])
async def admin_kpis(http_request: Request):
    return http_request.app.state.trace_store.kpi_summary()


# ---------------------------------------------------------------------------
# Cases endpoint
# ---------------------------------------------------------------------------
class _CreateCaseRequest(_BaseModel):
    order_id:    str
    reason:      str
    customer_id: str
    amount_inr:  float | None = None
    priority:    str = "medium"


@app.post("/cases", status_code=status.HTTP_201_CREATED, tags=["cases"],
          summary="Create a new CRM support case")
async def create_case(body: _CreateCaseRequest, http_request: Request):
    crm: CrmTool = http_request.app.state.crm_tool
    case = await crm.create_case(
        customer_id=body.customer_id,
        order_id=body.order_id.strip().upper(),
        reason=body.reason,
        amount_inr=body.amount_inr,
        trace_id=f"api-{body.order_id.strip().upper()}",
        priority=body.priority,
    )
    return case


# ---------------------------------------------------------------------------
# KB search endpoint
# ---------------------------------------------------------------------------
@app.get("/kb/search", tags=["kb"], summary="Search knowledge base articles by tags")
async def kb_search(tags: str, http_request: Request):
    """
    Search KB articles by comma-separated tags.
    Example: GET /kb/search?tags=return,refund
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    kb: KbTool = http_request.app.state.kb_tool
    articles = await kb.search(tags=tag_list)
    return {"articles": articles}


# ---------------------------------------------------------------------------
# Session cleanup
# ---------------------------------------------------------------------------
@app.delete(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Clear server-side session history",
    tags=["ops"],
)
async def delete_session(session_id: str, http_request: Request):
    checkpointer = http_request.app.state.checkpointer
    try:
        if hasattr(checkpointer, "storage"):
            checkpointer.storage.pop(session_id, None)
    except Exception:
        pass
    return {"cleared": True}


# ---------------------------------------------------------------------------
# Auth models
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
