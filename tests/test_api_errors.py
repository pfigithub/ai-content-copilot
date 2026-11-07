import hashlib

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import get_ai_provider
from app.exceptions import (
    AIProviderUnavailableError,
    URLFetchError,
)
from app.main import app


TEST_TOKEN = "api-errors-test-token"
TEST_USER_ID = "api_errors_test_user"

client = TestClient(app)


@pytest.fixture
def authenticated_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Configure a valid API key for authenticated API error tests."""
    digest = hashlib.sha256(
        TEST_TOKEN.encode("utf-8")
    ).hexdigest()

    monkeypatch.setattr(
        settings,
        "api_key_hashes",
        {digest: TEST_USER_ID},
    )

    return client


def auth_headers() -> dict[str, str]:
    """Return the Bearer authentication header."""
    return {
        "Authorization": f"Bearer {TEST_TOKEN}",
    }


def test_ai_provider_unavailable_returns_503(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Translate provider unavailability into the standard 503 response."""

    def raise_error():
        raise AIProviderUnavailableError()

    monkeypatch.setitem(
        app.dependency_overrides,
        get_ai_provider,
        raise_error,
    )

    response = authenticated_client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": "Test content.",
            "task": "summarize",
        },
        headers=auth_headers(),
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == (
        "AI_PROVIDER_UNAVAILABLE"
    )
    assert response.json()["error"]["message"] == (
        "The AI service is temporarily unavailable."
    )


def test_invalid_request_returns_standard_422(
    authenticated_client: TestClient,
):
    """Return validation errors after the request is authenticated."""

    response = authenticated_client.post(
        "/api/v1/content/analyze-content",
        json={},
        headers=auth_headers(),
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_url_fetch_failure_returns_502(
    authenticated_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
):
    """Translate URL retrieval failures into the standard 502 response."""

    async def fake_normalize_content(request):
        raise URLFetchError()

    monkeypatch.setattr(
        "app.api.v1.endpoints.content.normalize_content",
        fake_normalize_content,
    )

    response = authenticated_client.post(
        "/api/v1/content/analyze-content",
        json={
            "url": "https://example.com/failure",
            "task": "summarize",
        },
        headers=auth_headers(),
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "URL_FETCH_FAILED"
    assert response.json()["error"]["message"] == (
        "The requested URL could not be retrieved."
    )