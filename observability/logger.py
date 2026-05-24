"""
observability/logger.py
========================
Structured logging configuration for AtlasCare.

Responsibility
--------------
  Configure the Python logging system once at application startup
  with:
    - JSON-structured output for log aggregation pipelines
      (Datadog, Splunk, CloudWatch, ELK)
    - Human-readable fallback for local development
    - Consistent field set on every log record:
        timestamp, level, logger, message, trace_id (when available)
    - Log level controllable via LOG_LEVEL env var
    - Sensitive field redaction before emission

Design principles
-----------------
- configure_logging() is idempotent — safe to call multiple times.
- JSON format in production; plain text when LOG_FORMAT=text.
- No third-party logging libraries required — stdlib only.
- Sensitive fields (api_key, password, token) are redacted
  automatically in the JSON formatter.
"""

import json
import logging
import logging.config
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_LOG_LEVEL  = "INFO"
_DEFAULT_LOG_FORMAT = "json"   # "json" | "text"

# Fields that must never appear in log output
_SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "password", "passwd", "secret",
    "token", "access_token", "refresh_token", "authorization",
    "gemini_api_key",
})

_REDACTED = "***REDACTED***"


# ---------------------------------------------------------------------------
# JSON Formatter
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Every record emits:
      timestamp  : ISO 8601 UTC
      level      : DEBUG | INFO | WARNING | ERROR | CRITICAL
      logger     : dotted module path
      message    : formatted log message
      exc_info   : exception traceback (only when present)

    Extra fields passed via the `extra` kwarg on log calls are
    merged into the top-level JSON object after redaction.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "message": record.getMessage(),
        }

        # Exception info
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        # Merge any extra fields, redacting sensitive keys
        for key, value in record.__dict__.items():
            if key in _EXTRA_SKIP_KEYS:
                continue
            if key.startswith("_"):
                continue
            if key in _LOG_RECORD_BUILTIN_KEYS:
                continue
            payload[key] = _redact_value(key, value)

        return json.dumps(payload, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text Formatter (local dev)
# ---------------------------------------------------------------------------
class TextFormatter(logging.Formatter):
    """
    Human-readable formatter for local development.
    Format: LEVEL     logger:lineno  message
    """
    _FMT = "%(asctime)s  %(levelname)-8s  %(name)s:%(lineno)d  %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_configured = False


def configure_logging() -> None:
    """
    Configure the root logger for AtlasCare.
    Idempotent — subsequent calls are no-ops.

    Environment variables
    ---------------------
    LOG_LEVEL   : DEBUG | INFO | WARNING | ERROR  (default: INFO)
    LOG_FORMAT  : json | text                     (default: json)
    """
    global _configured
    if _configured:
        return

    level_name  = os.getenv("LOG_LEVEL",  _DEFAULT_LOG_LEVEL).upper()
    log_format  = os.getenv("LOG_FORMAT", _DEFAULT_LOG_FORMAT).lower()

    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    if log_format == "text":
        handler.setFormatter(TextFormatter())
    else:
        handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove any handlers added by previous configure calls or
    # by libraries that call basicConfig before us
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Suppress overly verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _configured = True

    logging.getLogger(__name__).info(
        "Logging configured | level=%s | format=%s",
        level_name,
        log_format,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _redact_value(key: str, value: Any) -> Any:
    """
    Redact a field value if the key matches any sensitive pattern.
    Operates on the key name only — does not inspect values.
    """
    key_lower = key.lower()
    if any(s in key_lower for s in _SENSITIVE_KEYS):
        return _REDACTED
    return value


# Built-in LogRecord attributes to skip when merging extra fields
_LOG_RECORD_BUILTIN_KEYS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
})

_EXTRA_SKIP_KEYS = frozenset({
    "color_message",   # uvicorn internal
})