import pytest

from fastapi.testclient import TestClient

from app.dependencies import get_ai_provider
from app.main import app
from app.providers.mock_provider import MockAIProvider


@pytest.fixture
def client():

    app.dependency_overrides[
        get_ai_provider
    ] = lambda: MockAIProvider()

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_analyze_content_with_text(client):

    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": (
                "Artificial intelligence is changing "
                "how people create and process content."
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "text"
    assert body["title"] == "Content analysis result"

    assert body["summary"].startswith(
        "Mock analysis:"
    )

    assert body["importance"] == 50
    assert body["category"] == "other"


def test_analyze_content_with_url(
    client,
    monkeypatch,
):

    async def fake_fetch_url_content(
        url: str,
    ):
        assert url == "https://example.com/"

        return (
            "Example page content for URL testing."
        )

    monkeypatch.setattr(
        "app.services.input_service.fetch_url_content",
        fake_fetch_url_content,
    )

    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "url": "https://example.com/",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "url"

    assert (
        "Example page content"
        in body["summary"]
    )

    assert body["importance"] == 50
    assert body["category"] == "other"


def test_analyze_pdf(
    client,
    monkeypatch,
):

    def fake_extract_text_from_pdf(
        file_path,
    ):
        return (
            "PDF test content for "
            "the content service."
        )

    monkeypatch.setattr(
        "app.api.v1.endpoints.content.extract_text_from_pdf",
        fake_extract_text_from_pdf,
    )

    response = client.post(
        "/api/v1/content/analyze-pdf",
        files={
            "file": (
                "test.pdf",
                b"fake pdf bytes",
                "application/pdf",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "pdf"
    assert body["title"] == "Content analysis result"
    assert body["importance"] == 50