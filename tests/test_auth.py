
import hashlib

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config import Settings, settings
from app.dependencies import get_current_user


TEST_TOKEN = "unit-test-api-key-for-authentication"
TEST_USER_ID = "test_user"


@pytest.fixture
def authenticated_client(monkeypatch):
    """Create a protected test endpoint with one configured API key."""

    digest = hashlib.sha256(
        TEST_TOKEN.encode("utf-8")
    ).hexdigest()

    monkeypatch.setattr(
        settings,
        "api_key_hashes",
        {digest: TEST_USER_ID},
    )

    application = FastAPI()

    @application.get("/protected")
    def protected_endpoint(
        current_user: str = Depends(get_current_user),
    ):
        return {"user_id": current_user}

    return TestClient(application)


def test_valid_api_key_resolves_server_side_user(
    authenticated_client,
):
    """A valid API key must resolve to its configured user."""

    response = authenticated_client.get(
        "/protected",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )

    assert response.status_code == 200
    assert response.json() == {"user_id": TEST_USER_ID}


def test_missing_api_key_is_rejected(
    authenticated_client,
):
    """Requests without an Authorization header must be rejected."""

    response = authenticated_client.get("/protected")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_x_user_id_header_cannot_authenticate(
    authenticated_client,
):
    """A caller-supplied user ID must never establish identity."""

    response = authenticated_client.get(
        "/protected",
        headers={"X-User-ID": "admin"},
    )

    assert response.status_code == 401


def test_unknown_api_key_is_rejected(
    authenticated_client,
):
    """An unregistered API key must not authenticate."""

    response = authenticated_client.get(
        "/protected",
        headers={"Authorization": "Bearer unknown-api-key"},
    )

    assert response.status_code == 401


def test_authentication_fails_when_no_keys_are_configured(
    authenticated_client,
    monkeypatch,
):
    """Protected endpoints must fail closed without API-key configuration."""

    monkeypatch.setattr(settings, "api_key_hashes", {})

    response = authenticated_client.get(
        "/protected",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )

    assert response.status_code == 503


def test_settings_reject_invalid_sha256_digest():
    """Configuration must reject malformed API-key digests."""

    with pytest.raises(ValidationError):
        Settings(
            api_key_hashes={
                "not-a-valid-sha256-digest": TEST_USER_ID,
            }
        )


def test_settings_reject_invalid_user_id():
    """Configuration must reject unsafe server-side user IDs."""

    valid_digest = hashlib.sha256(
        TEST_TOKEN.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValidationError):
        Settings(
            api_key_hashes={
                valid_digest: "invalid user id",
            }
        )
