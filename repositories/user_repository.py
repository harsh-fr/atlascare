import json
import logging
import os
import threading
from typing import Any

from utils.file_ops import atomic_json_write

logger = logging.getLogger(__name__)

_DEFAULT_USERS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "users.json"
)


class UserRepository:
    """Thread-safe CRUD access for the users.json credential store."""

    def __init__(self, path: str | None = None) -> None:
        self._path = os.path.abspath(
            path or os.getenv("USERS_DATA_PATH", _DEFAULT_USERS_PATH)
        )
        self._lock = threading.Lock()
        logger.debug("UserRepository init | path=%s", self._path)

    def _find_by_field(self, field: str, value: str) -> dict[str, Any] | None:
        value_lower = value.lower()
        for user in self._load().get("users", []):
            if user.get(field, "").lower() == value_lower:
                return user
        return None

    def get_by_username(self, username: str) -> dict[str, Any] | None:
        return self._find_by_field("username", username)

    def get_by_email(self, email: str) -> dict[str, Any] | None:
        return self._find_by_field("email", email)

    def username_exists(self, username: str) -> bool:
        return self.get_by_username(username) is not None

    def create(self, user: dict[str, Any]) -> bool:
        with self._lock:
            data = self._load()
            data.setdefault("users", []).append(user)
            self._save(data)
        return True

    def update_password(self, username: str, new_hash: str) -> bool:
        with self._lock:
            data = self._load()
            for user in data.get("users", []):
                if user.get("username", "").lower() == username.lower():
                    user["password_hash"] = new_hash
                    self._save(data)
                    return True
        return False

    def _load(self) -> dict[str, Any]:
        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {"users": []}
        except Exception as exc:
            logger.error("UserRepository._load error: %s", exc)
            return {"users": []}

    def _save(self, data: dict[str, Any]) -> None:
        atomic_json_write(self._path, data)
