import pytest

from fastapi.testclient import TestClient

from app.dependencies import (
    get_ai_provider,
)

from app.main import app

from app.providers.mock_provider import (
    MockAIProvider,
)


@pytest.fixture
def client():

    app.dependency_overrides[
        get_ai_provider
    ] = lambda: MockAIProvider()

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_analyze_content_with_summarize_task(
    client,
):

    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": (
                "Artificial intelligence is "
                "changing how people create content."
            ),
            "task": "summarize",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "text"
    assert body["title"] == (
        "Content analysis result"
    )
    assert "summary" in body
    assert "importance" in body
    assert "analysis" not in body
    assert "ideas" not in body


def test_analyze_content_with_analyze_task(
    client,
):

    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": (
                "Artificial intelligence is "
                "changing how people work."
            ),
            "task": "analyze",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "text"
    assert "analysis" in body
    assert "importance" in body
    assert "summary" not in body
    assert "ideas" not in body


def test_analyze_content_with_ideas_task(
    client,
):

    response = client.post(
        "/api/v1/content/analyze-content",
        json={
            "text": (
                "Many content creators struggle "
                "to find new content ideas."
            ),
            "task": "ideas",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "text"
    assert "summary" in body
    assert "ideas" in body
    assert len(body["ideas"]) >= 1
    assert "importance" not in body
    assert "analysis" not in body


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
            "task": "summarize",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "url"
    assert "summary" in body
    assert "importance" in body


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
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "pdf"
    assert body["title"] == (
        "Content analysis result"
    )
    assert "summary" in body
    assert "importance" in body
    