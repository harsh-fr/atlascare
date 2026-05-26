"""
tests/test_auth.py
==================
Tests for authentication: AuthService unit tests and API endpoint tests.

Seed credentials (from conftest data_dir fixture):
  username=alice  password=Atlas@123  customer_id=CUST-001
  username=bob    password=Atlas@456  customer_id=CUST-002
"""

import hashlib
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from services.auth_service import AuthService, DUMMY_OTP, _hash
from repositories.user_repository import UserRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_repo(tmp_path: Path) -> UserRepository:
    users_file = tmp_path / "users.json"
    users_data = {
        "users": [
            {
                "user_id":       "USER-001",
                "username":      "alice",
                "email":         "alice@test.com",
                "password_hash": _hash("Atlas@123"),
                "customer_id":   "CUST-001",
                "created_at":    "2025-01-01T00:00:00Z",
            },
            {
                "user_id":       "USER-002",
                "username":      "bob",
                "email":         "bob@test.com",
                "password_hash": _hash("Atlas@456"),
                "customer_id":   "CUST-002",
                "created_at":    "2025-01-01T00:00:00Z",
            },
        ]
    }
    users_file.write_text(json.dumps(users_data, indent=2), encoding="utf-8")
    return UserRepository(str(users_file))


@pytest.fixture
def svc(tmp_path):
    return AuthService(repo=_make_repo(tmp_path))


# ===========================================================================
# AuthService — login
# ===========================================================================

class TestLogin:

    def test_login_success(self, svc):
        result = svc.login("alice", "Atlas@123")
        assert result.success is True
        assert result.session_id is not None
        assert result.customer_id == "CUST-001"
        assert result.error is None

    def test_login_session_id_encodes_customer(self, svc):
        result = svc.login("alice", "Atlas@123")
        assert "CUST001" in result.session_id

    def test_login_wrong_password(self, svc):
        result = svc.login("alice", "wrong")
        assert result.success is False
        assert result.error == "Invalid username or password."
        assert result.session_id is None

    def test_login_unknown_username(self, svc):
        result = svc.login("nobody", "Atlas@123")
        assert result.success is False
        assert result.error == "Invalid username or password."

    def test_login_case_insensitive_username(self, svc):
        result = svc.login("ALICE", "Atlas@123")
        assert result.success is True

    def test_login_empty_password(self, svc):
        result = svc.login("alice", "")
        assert result.success is False


# ===========================================================================
# AuthService — register
# ===========================================================================

class TestRegister:

    def test_register_success(self, svc):
        result = svc.register("charlie", "Secure@99", "charlie@test.com", "CUST-003")
        assert result.success is True
        assert result.message is not None

    def test_register_duplicate_username(self, svc):
        result = svc.register("alice", "NewPass@1", "new@test.com", "CUST-004")
        assert result.success is False
        assert "taken" in result.error.lower()

    def test_register_duplicate_email(self, svc):
        result = svc.register("newuser", "NewPass@1", "alice@test.com", "CUST-004")
        assert result.success is False
        assert "email" in result.error.lower()

    def test_register_short_password(self, svc):
        result = svc.register("newuser", "abc", "new@test.com", "CUST-004")
        assert result.success is False
        assert "6 characters" in result.error

    def test_registered_user_can_login(self, svc):
        svc.register("dave", "Dave@Pass1", "dave@test.com", "CUST-004")
        result = svc.login("dave", "Dave@Pass1")
        assert result.success is True
        assert result.customer_id == "CUST-004"

    def test_register_normalises_customer_id_to_uppercase(self, svc):
        svc.register("eve", "Eve@Pass99", "eve@test.com", "cust-005")
        result = svc.login("eve", "Eve@Pass99")
        assert result.customer_id == "CUST-005"


# ===========================================================================
# AuthService — forgot password (OTP flow)
# ===========================================================================

class TestForgotPassword:

    def test_request_otp_known_user(self, svc):
        result = svc.request_otp("alice")
        assert result.success is True
        assert DUMMY_OTP in result.message

    def test_request_otp_unknown_user_does_not_reveal(self, svc):
        result = svc.request_otp("ghost")
        assert result.success is True   # does not reveal absence

    def test_reset_with_correct_otp(self, svc):
        result = svc.reset_password("alice", DUMMY_OTP, "NewAtlas@99")
        assert result.success is True
        assert svc.login("alice", "NewAtlas@99").success is True

    def test_reset_with_wrong_otp(self, svc):
        result = svc.reset_password("alice", "0000", "NewPass@1")
        assert result.success is False
        assert "Invalid OTP" in result.error

    def test_reset_with_short_password(self, svc):
        result = svc.reset_password("alice", DUMMY_OTP, "abc")
        assert result.success is False
        assert "6 characters" in result.error

    def test_reset_unknown_username(self, svc):
        result = svc.reset_password("nobody", DUMMY_OTP, "NewPass@99")
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_old_password_invalid_after_reset(self, svc):
        svc.reset_password("alice", DUMMY_OTP, "NewAtlas@99")
        assert svc.login("alice", "Atlas@123").success is False


# ===========================================================================
# UserRepository — direct unit tests
# ===========================================================================

class TestUserRepository:

    def test_get_by_username(self, tmp_path):
        repo = _make_repo(tmp_path)
        user = repo.get_by_username("alice")
        assert user is not None
        assert user["customer_id"] == "CUST-001"

    def test_get_by_username_missing(self, tmp_path):
        repo = _make_repo(tmp_path)
        assert repo.get_by_username("nobody") is None

    def test_get_by_email(self, tmp_path):
        repo = _make_repo(tmp_path)
        user = repo.get_by_email("bob@test.com")
        assert user is not None
        assert user["username"] == "bob"

    def test_username_exists(self, tmp_path):
        repo = _make_repo(tmp_path)
        assert repo.username_exists("alice") is True
        assert repo.username_exists("unknown") is False

    def test_create_and_retrieve(self, tmp_path):
        repo = _make_repo(tmp_path)
        new_user = {
            "user_id": "USER-NEW", "username": "zoe",
            "email": "zoe@test.com", "password_hash": _hash("pass"),
            "customer_id": "CUST-099", "created_at": "2025-01-01T00:00:00Z",
        }
        repo.create(new_user)
        assert repo.get_by_username("zoe") is not None

    def test_update_password(self, tmp_path):
        repo = _make_repo(tmp_path)
        new_hash = _hash("brandnew")
        result = repo.update_password("alice", new_hash)
        assert result is True
        user = repo.get_by_username("alice")
        assert user["password_hash"] == new_hash

    def test_update_password_unknown_user(self, tmp_path):
        repo = _make_repo(tmp_path)
        assert repo.update_password("nobody", _hash("x")) is False


# ===========================================================================
# API endpoint tests (via FastAPI TestClient)
# ===========================================================================

class TestAuthAPI:

    def test_login_endpoint_success(self, client):
        resp = client.post("/auth/login",
                           json={"username": "alice", "password": "Atlas@123"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["session_id"] is not None
        assert data["customer_id"] == "CUST-001"

    def test_login_endpoint_failure(self, client):
        resp = client.post("/auth/login",
                           json={"username": "alice", "password": "wrong"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert data["error"] is not None

    def test_register_endpoint_success(self, client):
        resp = client.post("/auth/register", json={
            "username": "newuser", "password": "NewPass@1",
            "email": "newuser@test.com", "customer_id": "CUST-099",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_register_endpoint_duplicate(self, client):
        resp = client.post("/auth/register", json={
            "username": "alice", "password": "NewPass@1",
            "email": "other@test.com", "customer_id": "CUST-099",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_request_otp_endpoint(self, client):
        resp = client.post("/auth/request-otp",
                           json={"username": "alice"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert DUMMY_OTP in data["message"]

    def test_reset_password_endpoint_success(self, client):
        resp = client.post("/auth/reset-password", json={
            "username": "alice", "otp": DUMMY_OTP,
            "new_password": "NewAtlas@99",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_reset_password_endpoint_wrong_otp(self, client):
        resp = client.post("/auth/reset-password", json={
            "username": "alice", "otp": "1234",
            "new_password": "NewAtlas@99",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is False

    def test_session_delete_endpoint(self, client):
        resp = client.delete("/session/sess-cust001")
        assert resp.status_code == 200
        assert resp.json()["cleared"] is True
