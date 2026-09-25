from fastapi.testclient import (
    TestClient,
)

from app.dependencies import get_ai_provider
from app.exceptions import (
    AIProviderUnavailableError,
    URLFetchError,
)

from app.main import app


client = TestClient(
    app
)


def test_ai_provider_unavailable_returns_503():
    def raise_error():
        raise AIProviderUnavailableError()

    app.dependency_overrides[get_ai_provider] = raise_error

    try:
        response = client.post(
            "/api/v1/content/analyze-content",
            json={
                "text": "Test content.",
                "task": "summarize",
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AI_PROVIDER_UNAVAILABLE"
    assert (
        response.json()["error"]["message"]
        == "The AI service is temporarily unavailable."
    )

def test_invalid_request_returns_standard_422():

    response = client.post(
        "/api/v1/content/analyze-content",
        json={},
    )

    assert (
        response.status_code
        == 422
    )

    assert (
        response.json()
        ["error"]
        ["code"]
        == "VALIDATION_ERROR"
    )


def test_url_fetch_failure_returns_502(monkeypatch):
    async def fake_normalize_content(request):
        raise URLFetchError()

    monkeypatch.setattr(
        "app.api.v1.endpoints.content.normalize_content",
        fake_normalize_content,
    )

    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "url": "https://example.com/failure",
            "task": "summarize",
        },
    )

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "URL_FETCH_FAILED"
    assert (
        response.json()["error"]["message"]
        == "The requested URL could not be retrieved."
    )