
import hashlib

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import content as content_endpoint
from app.config import settings
from app.dependencies import get_ai_provider
from app.main import app
from app.providers.mock_provider import MockAIProvider


TEST_TOKEN = "content-api-test-token"
TEST_USER_ID = "content_test_user"


def auth_headers() -> dict[str, str]:
    """Build the Bearer authentication header for content API tests."""
    return {
        "Authorization": f"Bearer {TEST_TOKEN}"
    }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Configure a deterministic provider and authenticated test identity."""
    token_digest = hashlib.sha256(
        TEST_TOKEN.encode("utf-8")
    ).hexdigest()

    monkeypatch.setattr(
        settings,
        "api_key_hashes",
        {
            token_digest: TEST_USER_ID,
        },
    )

    app.dependency_overrides[get_ai_provider] = (
        lambda: MockAIProvider()
    )

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


def test_analyze_content_with_summarize_task(client: TestClient):
    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": "Artificial intelligence is changing how people create content.",
            "task": "summarize",
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "text"
    assert body["title"] == "Content analysis result"
    assert "summary" in body
    assert "importance" in body
    assert "analysis" not in body
    assert "ideas" not in body


def test_analyze_content_with_analyze_task(client: TestClient):
    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": "Artificial intelligence is changing how people work.",
            "task": "analyze",
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "text"
    assert "analysis" in body
    assert "importance" in body


def test_analyze_content_with_ideas_task(client: TestClient):
    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": "Many content creators struggle to find new content ideas.",
            "task": "ideas",
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "text"
    assert "summary" in body
    assert "ideas" in body
    assert len(body["ideas"]) >= 1


def test_analyze_content_with_url(
    client: TestClient,
    monkeypatch,
):
    async def fake_fetch_url_content(url: str):
        assert url == "https://example.com/"
        return "Example page content for URL testing."

    monkeypatch.setattr(
        "app.services.input_service.fetch_url_content",
        fake_fetch_url_content,
    )

    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "url": "https://example.com/",
            "task": "summarize",
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "url"
    assert "summary" in body
    assert "importance" in body


def test_analyze_pdf(
    client: TestClient,
    monkeypatch,
):
    def fake_extract_text_from_pdf(file_path):
        return (
            "PDF test content for the content service.",
            [
                {
                    "page": 0,
                    "text": "PDF test content for the content service.",
                }
            ],
        )

    monkeypatch.setattr(
        content_endpoint,
        "extract_text_from_pdf",
        fake_extract_text_from_pdf,
    )

    response = client.post(
        "/api/v1/content/analyze-pdf",
        data={
            "task": "summarize",
        },
        files={
            "file": (
                "test.pdf",
                b"fake pdf bytes",
                "application/pdf",
            )
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "pdf"
    assert body["title"] == "Content analysis result"
    assert "summary" in body
    assert "importance" in body


def test_output_guardrail_receives_complete_ideas_response(
    client: TestClient,
    monkeypatch,
):
    """All generated fields, including ideas, must be scanned before returning."""
    captured = {}

    class CapturingOutputGuardrail:
        async def verify_safe_output(self, text: str) -> None:
            captured["text"] = text

    monkeypatch.setattr(
        content_endpoint,
        "OutputGuardrail",
        CapturingOutputGuardrail,
    )

    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": "Many content creators need practical content ideas.",
            "task": "ideas",
        },
        headers=auth_headers(),
    )

    assert response.status_code == 200

    assert '"title"' in captured["text"]
    assert '"summary"' in captured["text"]
    assert '"ideas"' in captured["text"]
    assert '"category"' in captured["text"]


def test_content_endpoints_require_authentication(
    client: TestClient,
):
    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": "This request has no authentication.",
        },
    )

    assert response.status_code == 401


def test_legacy_user_id_header_does_not_authenticate(
    client: TestClient,
):
    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": "This request uses a legacy header.",
        },
        headers={
            "X-User-ID": TEST_USER_ID,
        },
    )

    assert response.status_code == 401
