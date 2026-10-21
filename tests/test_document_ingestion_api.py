from fastapi.testclient import TestClient

from app.api.v1.endpoints import ingestion as ingestion_endpoint

from app.api.v1.endpoints.ingestion import (
    get_document_ingestion_service,
)

from app.main import create_app

from app.models.schemas import (
    DocumentIngestionResponse,
    NormalizedContent,
)


class FakeIngestionService:
    """Capture incoming content without calling Ollama or Qdrant."""

    def __init__(self):
        self.received = []

    async def ingest(self, content):
        self.received.append(content)

        return DocumentIngestionResponse(
            source_type=content.source_type,
            source=content.source,
            chunks_created=1,
            document_ids=["test-document-1"],
        )


def create_test_app(service):
    """Create an isolated app and inject a fake ingestion service."""

    application = create_app(
        use_in_memory=True
    )

    application.dependency_overrides[
        get_document_ingestion_service
    ] = lambda: service

    return application


def test_ingest_text():
    service = FakeIngestionService()
    application = create_test_app(service)

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            json={
                "text": "Test content for document ingestion.",
                "source": "test-guide.txt",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["source_type"] == "text"
    assert body["chunks_created"] == 1
    assert body["document_ids"] == [
        "test-document-1"
    ]
    assert body["source"] == "test-guide.txt"
    assert service.received[0].source == "test-guide.txt"

    assert len(service.received) == 1

    assert service.received[0].content == (
        "Test content for document ingestion."
    )


def test_ingest_url(monkeypatch):
    service = FakeIngestionService()
    application = create_test_app(service)

    async def fake_normalize_content(request):
        return NormalizedContent(
            content="Fetched content for testing.",
            source_type="url",
            source=str(request.url),
            source_language="en",
        )

    monkeypatch.setattr(
        ingestion_endpoint,
        "normalize_content",
        fake_normalize_content,
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            json={
                "url": "https://example.com/article"
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["source_type"] == "url"
    assert body["source"] == (
        "https://example.com/article"
    )

    assert service.received[0].content == (
        "Fetched content for testing."
    )


def test_ingest_pdf(monkeypatch):
    service = FakeIngestionService()
    application = create_test_app(service)

    monkeypatch.setattr(
        ingestion_endpoint,
        "extract_text_from_pdf",
        lambda _path: "متن آزمایشی برای ورود PDF",
    )

    monkeypatch.setattr(
        ingestion_endpoint,
        "detect_language",
        lambda _text: "fa",
    )

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/documents/ingest-pdf",
            files={
                "file": (
                    "sample.pdf",
                    b"%PDF-1.4 test content",
                    "application/pdf",
                )
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["source_type"] == "pdf"
    assert body["source"] == "sample.pdf"
    assert body["chunks_created"] == 1

    assert service.received[0].content == (
        "متن آزمایشی برای ورود PDF"
    )


def test_ingest_pdf_rejects_non_pdf_file():
    service = FakeIngestionService()
    application = create_test_app(service)

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/documents/ingest-pdf",
            files={
                "file": (
                    "notes.txt",
                    b"not a PDF",
                    "text/plain",
                )
            },
        )

    assert response.status_code == 415
    assert service.received == []


def test_ingest_rejects_blank_text():
    service = FakeIngestionService()
    application = create_test_app(service)

    with TestClient(application) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            json={
                "text": "   "
            },
        )

    assert response.status_code == 422
    assert service.received == []