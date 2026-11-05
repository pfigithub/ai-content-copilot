
import asyncio

import pytest
from langchain_core.documents import Document

from app.models.schemas import NormalizedContent
from app.services.document_ingestion_service import (
    DocumentIngestionService,
)


class FakeStore:
    """Simulate tenant-scoped vector storage and stale-chunk cleanup."""

    def __init__(self):
        self.documents_by_id: dict[str, Document] = {}

    @property
    def documents(self) -> list[Document]:
        return list(self.documents_by_id.values())

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
        *,
        user_id: str,
    ) -> list[str]:
        """Reject cross-tenant writes and simulate deterministic upserts."""

        if not user_id:
            raise ValueError("user_id is required")

        if ids is None:
            raise ValueError("Deterministic IDs are required in this test.")

        if len(ids) != len(documents):
            raise ValueError("IDs and documents must have matching lengths.")

        for document_id, document in zip(ids, documents):
            if document.metadata.get("user_id") != user_id:
                raise ValueError("Tenant mismatch.")

            self.documents_by_id[document_id] = document

        return list(ids)

    def delete_chunks_after(
        self,
        source_id: str,
        last_chunk_index: int,
        *,
        user_id: str,
    ) -> None:
        """Remove trailing chunks only for the specified tenant and source."""

        obsolete_ids = [
            document_id
            for document_id, document in self.documents_by_id.items()
            if (
                document.metadata.get("user_id") == user_id
                and document.metadata.get("source_id") == source_id
                and document.metadata.get("chunk_index", 0)
                > last_chunk_index
            )
        ]

        for document_id in obsolete_ids:
            del self.documents_by_id[document_id]


def make_content(
    user_id: str | None = "tenant-a",
    source: str = "water-guide.txt",
    text: str | None = None,
) -> NormalizedContent:
    """Build a consistent test document for the selected tenant."""

    return NormalizedContent(
        content=text or (
            "Water management and efficient irrigation improve "
            "agricultural water use. "
        ) * 20,
        source_type="text",
        source=source,
        source_language="en",
        user_id=user_id,
    )


def test_ingestion_splits_and_stores_documents():
    """Verify chunk metadata and deterministic document IDs."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    result = asyncio.run(
        service.ingest(
            make_content(),
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    assert result.chunks_created > 1
    assert result.chunks_created == len(store.documents)
    assert result.source == "water-guide.txt"
    assert result.source_type == "text"
    assert len(result.document_ids) == len(store.documents)
    assert len(set(result.document_ids)) == len(result.document_ids)

    for index, document in enumerate(store.documents, start=1):
        assert document.metadata["source"] == "water-guide.txt"
        assert document.metadata["source_type"] == "text"
        assert document.metadata["user_id"] == "tenant-a"
        assert document.metadata["source_id"]
        assert document.metadata["chunk_index"] == index
        assert document.metadata["chunk_count"] == result.chunks_created


def test_reingestion_reuses_ids_for_same_tenant_and_source():
    """Repeated ingestion of one source must upsert rather than duplicate."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)
    content = make_content()

    first = asyncio.run(
        service.ingest(
            content,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    second = asyncio.run(
        service.ingest(
            content,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    assert second.document_ids == first.document_ids
    assert len(store.documents) == len(first.document_ids)


def test_same_source_name_is_isolated_between_tenants():
    """Identical source names must produce distinct tenant-scoped IDs."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    tenant_a = make_content(user_id="tenant-a")
    tenant_b = make_content(user_id="tenant-b")

    result_a = asyncio.run(
        service.ingest(
            tenant_a,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    result_b = asyncio.run(
        service.ingest(
            tenant_b,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    assert set(result_a.document_ids).isdisjoint(
        set(result_b.document_ids)
    )

    assert len(store.documents) == (
        result_a.chunks_created + result_b.chunks_created
    )

    owners = {
        document.metadata["user_id"]
        for document in store.documents
    }

    assert owners == {"tenant-a", "tenant-b"}


def test_shorter_reingestion_removes_only_its_own_trailing_chunks():
    """Replacing one tenant's source must not delete another tenant's data."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    original_a = make_content(
        user_id="tenant-a",
        text=(
            "Long document about water management and irrigation. "
        ) * 20,
    )

    original_b = make_content(
        user_id="tenant-b",
        text=(
            "Long document about water management and irrigation. "
        ) * 20,
    )

    result_a = asyncio.run(
        service.ingest(
            original_a,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    result_b = asyncio.run(
        service.ingest(
            original_b,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    assert result_a.chunks_created > 1
    assert result_b.chunks_created > 1

    shortened_a = make_content(
        user_id="tenant-a",
        text="Short replacement document.",
    )

    replacement = asyncio.run(
        service.ingest(
            shortened_a,
            chunk_size=80,
            chunk_overlap=10,
        )
    )

    tenant_a_documents = [
        document
        for document in store.documents
        if document.metadata["user_id"] == "tenant-a"
    ]

    tenant_b_documents = [
        document
        for document in store.documents
        if document.metadata["user_id"] == "tenant-b"
    ]

    assert replacement.chunks_created == 1
    assert len(tenant_a_documents) == 1
    assert len(tenant_b_documents) == result_b.chunks_created


def test_ingestion_rejects_missing_user_id():
    """Never store documents under a shared fallback tenant."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    with pytest.raises(
        ValueError,
        match="user_id is required",
    ):
        asyncio.run(
            service.ingest(
                make_content(user_id=None)
            )
        )

    assert store.documents == []


def test_ingestion_rejects_empty_content():
    """Reject blank content before storage is touched."""

    store = FakeStore()
    service = DocumentIngestionService(store=store)

    content = NormalizedContent(
        content="   ",
        source_type="text",
        user_id="tenant-a",
    )

    with pytest.raises(
        ValueError,
        match="Content cannot be empty",
    ):
        asyncio.run(
            service.ingest(content)
        )

    assert store.documents == []
