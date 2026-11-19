"""Integration tests for POST /auth/change-password.

These tests exercise the full HTTP stack via Starlette's TestClient
with in-memory repositories (APP_ENV=test), so no PostgreSQL or network
is required.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    """Build a TestClient with a known admin and a working JWT secret."""
    monkeypatch.setenv("APP_ENV", "test")

    from app.config import settings

    # A stable 64-byte secret so tokens round-trip predictably.
    monkeypatch.setattr(settings, "jwt_secret", "x" * 64)
    monkeypatch.setattr(settings, "first_superuser_username", "admin")
    monkeypatch.setattr(settings, "first_superuser_password", "adminpass123")

    from app.main import create_app

    app = create_app(use_in_memory=True)

    with TestClient(app) as test_client:
        yield test_client


def _login(client: TestClient, username: str, password: str):
    return client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )


def test_change_password_success_logs_out_other_sessions(client):
    # 1. Log in with the original password.
    login = _login(client, "admin", "adminpass123")
    assert login.status_code == 200
    tokens = login.json()
    access = tokens["access_token"]
    old_refresh = tokens["refresh_token"]

    # 2. Change the password.
    response = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {access}"},
        json={
            "current_password": "adminpass123",
            "new_password": "newadminpass456",
        },
    )
    assert response.status_code == 204

    # 3. Old password no longer logs in.
    assert _login(client, "admin", "adminpass123").status_code == 401

    # 4. New password works.
    assert _login(client, "admin", "newadminpass456").status_code == 200

    # 5. The pre-change refresh token was revoked (all sessions invalidated).
    refresh_attempt = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": old_refresh},
    )
    assert refresh_attempt.status_code == 401


def test_change_password_rejects_wrong_current_password(client):
    login = _login(client, "admin", "adminpass123")
    access = login.json()["access_token"]

    response = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {access}"},
        json={
            "current_password": "definitely-wrong",
            "new_password": "newadminpass456",
        },
    )
    assert response.status_code == 400

    # Password must remain unchanged: the original still works.
    assert _login(client, "admin", "adminpass123").status_code == 200


def test_change_password_rejects_same_as_current(client):
    login = _login(client, "admin", "adminpass123")
    access = login.json()["access_token"]

    response = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {access}"},
        json={
            "current_password": "adminpass123",
            "new_password": "adminpass123",
        },
    )
    assert response.status_code == 422


def test_change_password_requires_authentication(client):
    response = client.post(
        "/api/v1/auth/change-password",
        json={
            "current_password": "anything",
            "new_password": "newadminpass456",
        },
    )
    assert response.status_code == 401


def test_change_password_rejects_short_new_password(client):
    login = _login(client, "admin", "adminpass123")
    access = login.json()["access_token"]

    response = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {access}"},
        json={
            "current_password": "adminpass123",
            "new_password": "short",
        },
    )
    assert response.status_code == 422