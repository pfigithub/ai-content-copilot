
import hashlib

from fastapi.testclient import TestClient

from app.api.v1.endpoints import ingestion as ingestion_endpoint
from app.api.v1.endpoints.ingestion import get_document_ingestion_service
from app.config import settings
from app.main import create_app
from app.models.schemas import (
    DocumentIngestionResponse,
    NormalizedContent,
)


TEST_TOKEN = "document-ingestion-test-token"
TEST_USER_ID = "ingestion_test_user"


def auth_headers() -> dict[str, str]:
    """Build the Bearer authentication header for ingestion tests."""
    return {
        "Authorization": f"Bearer {TEST_TOKEN}"
    }


class FakeIngestionService:
    """Capture normalized ingestion requests without contacting Qdrant."""

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


def create_test_client(
    service,
    monkeypatch,
) -> TestClient:
    """Create an app with a mocked ingestion service and test API identity."""
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

    application = create_app(use_in_memory=True)

    application.dependency_overrides[
        get_document_ingestion_service
    ] = lambda: service

    return TestClient(application)


def test_ingest_text(monkeypatch):
    service = FakeIngestionService()

    with create_test_client(service, monkeypatch) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            json={
                "text": "Test content for document ingestion.",
                "source": "test-guide.txt",
            },
            headers=auth_headers(),
        )

    assert response.status_code == 200

    body = response.json()

    assert body["source_type"] == "text"
    assert body["chunks_created"] == 1
    assert service.received[0].content == (
        "Test content for document ingestion."
    )
    assert service.received[0].user_id == TEST_USER_ID


def test_ingest_url(monkeypatch):
    service = FakeIngestionService()

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

    with create_test_client(service, monkeypatch) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            json={
                "url": "https://example.com/article",
            },
            headers=auth_headers(),
        )

    assert response.status_code == 200

    body = response.json()

    assert body["source_type"] == "url"
    assert body["source"] == "https://example.com/article"
    assert service.received[0].user_id == TEST_USER_ID


def test_ingest_pdf(monkeypatch):
    service = FakeIngestionService()

    monkeypatch.setattr(
        ingestion_endpoint,
        "extract_text_from_pdf",
        lambda _path: (
            "متن آزمایشی برای ورود PDF",
            [
                {
                    "page": 0,
                    "text": "متن آزمایشی برای ورود PDF",
                }
            ],
        ),
    )

    monkeypatch.setattr(
        ingestion_endpoint,
        "detect_language",
        lambda _text: "fa",
    )

    with create_test_client(service, monkeypatch) as client:
        response = client.post(
            "/api/v1/documents/ingest-pdf",
            files={
                "file": (
                    "sample.pdf",
                    b"%PDF-1.4 test content",
                    "application/pdf",
                )
            },
            headers=auth_headers(),
        )

    assert response.status_code == 200

    body = response.json()

    assert body["source_type"] == "pdf"
    assert body["source"] == "sample.pdf"
    assert service.received[0].content == (
        "متن آزمایشی برای ورود PDF"
    )
    assert service.received[0].pages[0]["page"] == 0
    assert service.received[0].user_id == TEST_USER_ID


def test_ingest_pdf_rejects_non_pdf_file(monkeypatch):
    service = FakeIngestionService()

    with create_test_client(service, monkeypatch) as client:
        response = client.post(
            "/api/v1/documents/ingest-pdf",
            files={
                "file": (
                    "notes.txt",
                    b"not a PDF",
                    "text/plain",
                )
            },
            headers=auth_headers(),
        )

    assert response.status_code == 415
    assert service.received == []


def test_ingest_rejects_blank_text(monkeypatch):
    service = FakeIngestionService()

    with create_test_client(service, monkeypatch) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            json={
                "text": "   ",
            },
            headers=auth_headers(),
        )

    assert response.status_code == 422
    assert service.received == []


def test_ingestion_requires_authentication(monkeypatch):
    service = FakeIngestionService()

    with create_test_client(service, monkeypatch) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            json={
                "text": "Unauthenticated ingestion must be rejected.",
            },
        )

    assert response.status_code == 401
    assert service.received == []


def test_client_cannot_choose_document_owner(monkeypatch):
    service = FakeIngestionService()

    with create_test_client(service, monkeypatch) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            json={
                "text": "Try to set another owner.",
                "user_id": "some-other-user",
            },
            headers=auth_headers(),
        )

    assert response.status_code == 422
    assert service.received == []
