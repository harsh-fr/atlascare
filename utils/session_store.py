"""
utils/session_store.py
=======================
Session-to-customer identity mapping.

Responsibility
--------------
  Resolve an opaque session_id to an authenticated customer_id.
  This is the ownership enforcement entry point — every request
  must resolve to a known customer before any tool is called.

Design principles
-----------------
- The assignment has no authentication system, so this module
  implements a pragmatic simulation as required by the spec:
    "Implement a pragmatic simulation. session_id → customer_id mapping"
- Two resolution strategies:
    1. Explicit mapping : SESSION_MAP env var or sessions.json file
    2. Embedded pattern : session_id contains customer_id directly
       e.g. "sess-CUST001-abc123" → "CUST-001"
       This supports testing all customers without pre-configuring
       every session.
- Unknown sessions return None — callers treat this as auth failure.
- Resolution is deterministic and logged for audit.
- Thread-safe for async use — no mutable shared state after init.
"""

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_SESSIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "sessions.json"
)

# Pattern: sess-CUST001-<anything>  →  CUST-001
# Handles both "CUST001" (no hyphen) and "CUST-001" (with hyphen)
_EMBEDDED_PATTERN = re.compile(
    r"(?:^|[-_])CUST[-_]?(\d{3})(?:[-_]|$)",
    re.IGNORECASE,
)


class SessionStore:
    """
    Resolves session tokens to authenticated customer identities.

    Resolution order
    ----------------
    1. Explicit session map (sessions.json or SESSION_MAP env var)
    2. Embedded customer ID pattern in session_id string
    3. Return None (unknown session)
    """

    def __init__(self, sessions_path: str | None = None) -> None:
        self._explicit_map: dict[str, str] = {}
        self._load(sessions_path)
        logger.debug(
            "SessionStore loaded | explicit_sessions=%d",
            len(self._explicit_map),
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def resolve(self, session_id: str) -> str | None:
        """
        Resolve session_id to customer_id.

        Parameters
        ----------
        session_id : opaque session token from HTTP request

        Returns
        -------
        customer_id string if resolved, None if unknown.
        """
        if not session_id or not session_id.strip():
            logger.warning("SessionStore.resolve called with empty session_id.")
            return None

        # Strategy 1: explicit map lookup
        customer_id = self._explicit_map.get(session_id)
        if customer_id:
            logger.debug(
                "Session resolved via explicit map | session=%s | customer=%s",
                session_id,
                customer_id,
            )
            return customer_id

        # Strategy 2: embedded pattern extraction
        customer_id = self._extract_from_pattern(session_id)
        if customer_id:
            logger.debug(
                "Session resolved via pattern | session=%s | customer=%s",
                session_id,
                customer_id,
            )
            return customer_id

        logger.warning(
            "Session could not be resolved | session=%s",
            session_id,
        )
        return None

    def register(self, session_id: str, customer_id: str) -> None:
        """
        Register an explicit session → customer mapping at runtime.
        Used by tests and synthetic data setup.
        """
        self._explicit_map[session_id] = customer_id
        logger.debug(
            "SessionStore.register | session=%s | customer=%s",
            session_id,
            customer_id,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def _load(self, sessions_path: str | None) -> None:
        """
        Load explicit session mappings from:
          1. sessions.json file (if present)
          2. SESSION_MAP env var (JSON string of {session_id: customer_id})
        """
        # Load from file
        path = os.path.abspath(
            sessions_path
            or os.getenv("SESSIONS_DATA_PATH", _DEFAULT_SESSIONS_PATH)
        )
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                sessions: list[dict[str, Any]] = data.get("sessions", [])
                for entry in sessions:
                    sid = entry.get("session_id")
                    cid = entry.get("customer_id")
                    if sid and cid:
                        self._explicit_map[sid] = cid
                logger.debug(
                    "SessionStore: loaded %d sessions from %s",
                    len(self._explicit_map),
                    path,
                )
            except Exception as exc:
                logger.warning(
                    "SessionStore: failed to load sessions file '%s': %s",
                    path,
                    exc,
                )

        # Override / supplement with SESSION_MAP env var
        session_map_env = os.getenv("SESSION_MAP")
        if session_map_env:
            try:
                env_map: dict[str, str] = json.loads(session_map_env)
                self._explicit_map.update(env_map)
                logger.debug(
                    "SessionStore: loaded %d sessions from SESSION_MAP env var",
                    len(env_map),
                )
            except json.JSONDecodeError as exc:
                logger.warning(
                    "SessionStore: SESSION_MAP env var is not valid JSON: %s",
                    exc,
                )

    @staticmethod
    def _extract_from_pattern(session_id: str) -> str | None:
        """
        Extract customer_id from embedded pattern in session_id.

        Examples
        --------
        "sess-CUST001-abc123"  → "CUST-001"
        "CUST-001-session"     → "CUST-001"
        "user-CUST042-xyz"     → "CUST-042"
        "random-string"        → None
        """
        match = _EMBEDDED_PATTERN.search(session_id)
        if match:
            number = match.group(1).zfill(3)
            return f"CUST-{number}"
        return None