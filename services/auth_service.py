"""
services/auth_service.py
========================
Authentication: login, register, forgot-password with dummy OTP.

Design notes
------------
- Passwords stored as SHA-256 hex digest (no salt — demo system).
- Forgot-password OTP is always 9999 (demo mode, never sent over network).
- Session IDs embed customer_id so SessionStore resolves them via its
  built-in embedded-pattern extraction without needing sessions.json edits.
"""

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone

from repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)

DUMMY_OTP = "9999"


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _make_session_id(customer_id: str) -> str:
    """
    Encode customer_id into session_id so SessionStore pattern-extracts it.
    CUST-001 → sess-CUST001-<8 random hex chars>
    """
    number = customer_id.split("-")[-1]
    return f"sess-CUST{number}-{secrets.token_hex(4)}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class AuthResult:
    success: bool
    error: str | None = None
    session_id: str | None = None
    customer_id: str | None = None
    message: str | None = None


class AuthService:
    """Handles user authentication, registration, and password reset."""

    def __init__(self, repo: UserRepository | None = None) -> None:
        self._repo = repo or UserRepository()

    def login(self, username: str, password: str) -> AuthResult:
        user = self._repo.get_by_username(username)
        if user is None or user["password_hash"] != _hash(password):
            return AuthResult(success=False, error="Invalid username or password.")

        customer_id = user["customer_id"]
        session_id  = _make_session_id(customer_id)
        logger.info("Login success | username=%s | customer=%s", username, customer_id)
        return AuthResult(success=True, session_id=session_id, customer_id=customer_id)

    def register(
        self,
        username: str,
        password: str,
        email: str,
        customer_id: str,
    ) -> AuthResult:
        if len(password) < 6:
            return AuthResult(success=False, error="Password must be at least 6 characters.")
        if self._repo.username_exists(username):
            return AuthResult(success=False, error="Username already taken.")
        if self._repo.get_by_email(email) is not None:
            return AuthResult(success=False, error="Email already registered.")

        user = {
            "user_id":       f"USER-{secrets.token_hex(3).upper()}",
            "username":      username.strip(),
            "email":         email.strip().lower(),
            "password_hash": _hash(password),
            "customer_id":   customer_id.strip().upper(),
            "created_at":    _now_iso(),
        }
        self._repo.create(user)
        logger.info("Registration | username=%s | customer=%s", username, customer_id)
        return AuthResult(success=True, message="Account created. Please log in.")

    def request_otp(self, username: str) -> AuthResult:
        """Always succeeds — dummy OTP is 9999 (never sent over network in demo)."""
        user = self._repo.get_by_username(username)
        if user is None:
            # Don't reveal whether username exists
            return AuthResult(
                success=True,
                message=f"If that account exists, an OTP has been sent. (Demo: OTP is **{DUMMY_OTP}**)",
            )
        email = user.get("email", "")
        masked = f"{email[:2]}***{email[email.find('@'):]}" if "@" in email else "your email"
        return AuthResult(
            success=True,
            message=f"OTP sent to {masked}. (Demo mode: OTP is **{DUMMY_OTP}**)",
        )

    def reset_password(self, username: str, otp: str, new_password: str) -> AuthResult:
        if otp.strip() != DUMMY_OTP:
            return AuthResult(success=False, error="Invalid OTP. Please try again.")
        if len(new_password) < 6:
            return AuthResult(success=False, error="Password must be at least 6 characters.")
        if self._repo.get_by_username(username) is None:
            return AuthResult(success=False, error="Username not found.")

        self._repo.update_password(username, _hash(new_password))
        logger.info("Password reset | username=%s", username)
        return AuthResult(success=True, message="Password reset successfully. Please log in.")
