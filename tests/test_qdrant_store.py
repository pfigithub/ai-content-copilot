
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from app.retrieval.qdrant_store import QdrantStore


def make_document(
    user_id: str = "tenant-a",
    source_id: str = "source-123",
    chunk_index: int = 1,
) -> Document:
    """Create a document with the metadata required by tenant isolation."""

    return Document(
        page_content="Test content.",
        metadata={
            "user_id": user_id,
            "source_id": source_id,
            "source": "test.txt",
            "source_type": "text",
            "chunk_index": chunk_index,
            "chunk_count": 1,
        },
    )


def patch_qdrant_dependencies(monkeypatch):
    """Replace external Qdrant and embedding dependencies with test doubles."""

    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def collection_exists(self, collection_name):
            return True

        def delete(self, collection_name, points_selector, wait=True):
            captured["delete"] = {
                "collection_name": collection_name,
                "points_selector": points_selector,
                "wait": wait,
            }

    class FakeEmbeddingProvider:
        def __init__(self):
            self.embeddings = object()

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantClient",
        FakeClient,
    )
    monkeypatch.setattr(
        "app.retrieval.qdrant_store.OllamaEmbeddingProvider",
        FakeEmbeddingProvider,
    )

    return captured


def test_qdrant_store_adds_tenant_owned_documents(monkeypatch):
    """Store documents only when their metadata matches the tenant."""

    captured = patch_qdrant_dependencies(monkeypatch)

    class FakeVectorStore:
        @classmethod
        def from_documents(cls, **kwargs):
            captured["store"] = kwargs
            return cls()

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantVectorStore",
        FakeVectorStore,
    )

    store = QdrantStore()
    documents = [make_document()]

    ids = store.add_documents(
        documents,
        user_id="tenant-a",
    )

    assert len(ids) == 1
    assert captured["store"]["collection_name"] == "content_copilot"
    assert captured["store"]["documents"] == documents
    assert captured["store"]["ids"] == ids


def test_qdrant_store_rejects_document_owned_by_another_tenant(
    monkeypatch,
):
    """A tenant must not insert a document carrying another tenant ID."""

    patch_qdrant_dependencies(monkeypatch)

    store = QdrantStore()

    with pytest.raises(ValueError, match="does not belong"):
        store.add_documents(
            [make_document(user_id="tenant-b")],
            user_id="tenant-a",
        )


def test_qdrant_store_rejects_missing_source_id(monkeypatch):
    """Every stored chunk must identify its source document."""

    patch_qdrant_dependencies(monkeypatch)

    store = QdrantStore()
    document = make_document()
    document.metadata.pop("source_id")

    with pytest.raises(ValueError, match="source_id"):
        store.add_documents(
            [document],
            user_id="tenant-a",
        )


def test_qdrant_store_deletes_only_tenant_owned_trailing_chunks(
    monkeypatch,
):
    """Stale-chunk cleanup must include the tenant ownership filter."""

    captured = patch_qdrant_dependencies(monkeypatch)
    store = QdrantStore(collection_name="test_collection")

    store.delete_chunks_after(
        source_id="source-123",
        last_chunk_index=2,
        user_id="tenant-a",
    )

    deletion = captured["delete"]
    conditions = deletion["points_selector"].must

    assert deletion["wait"] is True
    assert len(conditions) == 3

    assert conditions[0].key == "metadata.user_id"
    assert conditions[0].match.value == "tenant-a"

    assert conditions[1].key == "metadata.source_id"
    assert conditions[1].match.value == "source-123"

    assert conditions[2].key == "metadata.chunk_index"
    assert conditions[2].range.gt == 2


@pytest.mark.anyio
async def test_qdrant_store_filters_retrieval_by_tenant(monkeypatch):
    """Search results must be restricted to the requested tenant."""

    patch_qdrant_dependencies(monkeypatch)

    expected_results = [
        (make_document(), 0.62)
    ]

    vector_store = MagicMock()
    vector_store.asimilarity_search_with_score = AsyncMock(
        return_value=expected_results
    )

    vector_store_factory = MagicMock()
    vector_store_factory.from_existing_collection.return_value = vector_store

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantVectorStore",
        vector_store_factory,
    )

    store = QdrantStore(
        collection_name="test_collection"
    )

    results = await store.similarity_search_with_score(
        query="A test retrieval query.",
        k=3,
        score_threshold=0.4,
        user_id="tenant-a",
    )

    assert results == expected_results

    expected_filter = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value="tenant-a"),
            )
        ]
    )

    vector_store.asimilarity_search_with_score.assert_awaited_once_with(
        "A test retrieval query.",
        k=3,
        score_threshold=0.4,
        filter=expected_filter,
    )


@pytest.mark.anyio
async def test_qdrant_store_rejects_missing_user_id(monkeypatch):
    """Retrieval without a tenant identity must fail closed."""

    patch_qdrant_dependencies(monkeypatch)

    store = QdrantStore(
        collection_name="test_collection"
    )

    with pytest.raises(
        ValueError,
        match="user_id is required",
    ):
        await store.similarity_search_with_score(
            query="Test query.",
            user_id=None,
        )


def test_qdrant_store_rejects_missing_tenant_on_insert(monkeypatch):
    """Insertion without a tenant identity must fail closed."""

    patch_qdrant_dependencies(monkeypatch)

    store = QdrantStore()

    with pytest.raises(
        ValueError,
        match="user_id is required",
    ):
        store.add_documents(
            [make_document()],
            user_id="",
        )


def test_qdrant_store_rejects_missing_tenant_on_delete(monkeypatch):
    """Deletion without a tenant identity must fail closed."""

    patch_qdrant_dependencies(monkeypatch)

    store = QdrantStore()

    with pytest.raises(
        ValueError,
        match="user_id is required",
    ):
        store.delete_chunks_after(
            source_id="source-123",
            last_chunk_index=1,
            user_id="",
        )
