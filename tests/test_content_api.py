from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_analyze_content_with_text():
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


def test_analyze_content_with_url(monkeypatch):

    async def fake_fetch_url_content(url: str):
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
            "url": "https://example.com/"
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["input_type"] == "url"
    assert body["title"] == "Content analysis result"
    assert (
        "Example page content"
        in body["summary"]
    )
    assert body["importance"] == 50
    assert body["category"] == "other"