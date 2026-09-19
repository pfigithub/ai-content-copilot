from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_analyze_content_endpoint():
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
    assert body["summary"].startswith("Mock analysis:")
    assert body["importance"] == 50
    assert body["category"] == "other"