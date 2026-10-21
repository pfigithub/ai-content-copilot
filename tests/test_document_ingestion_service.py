import asyncio

from langchain_core.documents import Document

from app.models.schemas import NormalizedContent
from app.services.document_ingestion_service import (
    DocumentIngestionService,
)


class FakeStore:
    """Simulate Qdrant upserts and stale-chunk cleanup in memory."""

    def __init__(self):
        """Initialize the in-memory collection."""

        self.documents_by_id: dict[str, Document] = {}

    @property
    def documents(self) -> list[Document]:
        """Return the documents currently stored by the fake collection."""

        return list(self.documents_by_id.values())

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Simulate insertion or replacement using document IDs."""

        if ids is None:
            ids = [
                f"fake-document-{index}"
                for index in range(1, len(documents) + 1)
            ]

        for document_id, document in zip(ids, documents):
            self.documents_by_id[document_id] = document

        return list(ids)

    def delete_chunks_after(
        self,
        source_id: str,
        last_chunk_index: int,
    ) -> None:
        """Remove trailing chunks belonging to the specified source."""

        obsolete_ids = [
            document_id
            for document_id, document in self.documents_by_id.items()
            if (
                document.metadata.get("source_id") == source_id
                and document.metadata.get("chunk_index", 0)
                > last_chunk_index
            )
        ]

        for document_id in obsolete_ids:
            del self.documents_by_id[document_id]


def test_ingestion_splits_and_stores_documents():
    """Verify chunk creation, metadata preservation, and stable IDs."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    content = NormalizedContent(
        content=(
            "آبیاری قطره‌ای می‌تواند مصرف آب را کاهش دهد. "
            "مدیریت صحیح منابع آب به افزایش بهره‌وری کمک می‌کند. "
        ) * 20,
        source_type="text",
        source="agriculture-guide.txt",
        source_language="fa",
    )

    result = asyncio.run(
        service.ingest(
            content=content,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    assert result.chunks_created > 1
    assert result.chunks_created == len(store.documents)
    assert result.source == "agriculture-guide.txt"
    assert result.source_type == "text"
    assert len(result.document_ids) == len(store.documents)

    assert len(set(result.document_ids)) == len(
        result.document_ids
    )

    for index, document in enumerate(
        store.documents,
        start=1,
    ):
        assert document.metadata["source"] == (
            "agriculture-guide.txt"
        )

        assert document.metadata["source_type"] == "text"
        assert document.metadata["source_id"]
        assert document.metadata["chunk_index"] == index
        assert document.metadata["chunk_count"] == (
            result.chunks_created
        )


def test_reingestion_does_not_duplicate_identical_source():
    """Verify repeating an ingestion reuses existing document IDs."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    content = NormalizedContent(
        content=(
            "آبیاری قطره‌ای می‌تواند مصرف آب را کاهش دهد. "
        ) * 10,
        source_type="text",
        source="agriculture-guide.txt",
    )

    first_result = asyncio.run(
        service.ingest(
            content,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    first_ids = list(first_result.document_ids)

    second_result = asyncio.run(
        service.ingest(
            content,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    assert second_result.document_ids == first_ids
    assert len(store.documents) == len(first_ids)


def test_reingestion_removes_obsolete_trailing_chunks():
    """Verify shorter replacement content removes old trailing chunks."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    original_content = NormalizedContent(
        content=(
            "این یک راهنمای طولانی درباره مدیریت آب است. "
        ) * 20,
        source_type="text",
        source="water-guide.txt",
    )

    original_result = asyncio.run(
        service.ingest(
            original_content,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    assert original_result.chunks_created > 1

    updated_content = NormalizedContent(
        content="نسخه کوتاه‌شده راهنمای مدیریت آب.",
        source_type="text",
        source="water-guide.txt",
    )

    updated_result = asyncio.run(
        service.ingest(
            updated_content,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    assert updated_result.chunks_created == 1
    assert len(store.documents) == 1
    assert store.documents[0].page_content == (
        "نسخه کوتاه‌شده راهنمای مدیریت آب."
    )


def test_ingestion_rejects_empty_content():
    """Verify blank content is rejected before storage is touched."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    content = NormalizedContent(
        content="   ",
        source_type="text",
    )

    try:
        asyncio.run(service.ingest(content))
        raise AssertionError(
            "Expected empty content to be rejected."
        )
    except ValueError as exc:
        assert "Content cannot be empty" in str(exc)

    assert store.documents == []