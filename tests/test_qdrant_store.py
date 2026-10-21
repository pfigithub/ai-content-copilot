from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from langchain_core.documents import (
    Document,
)

from app.retrieval.qdrant_store import (
    QdrantStore,
)


def test_qdrant_store_adds_documents(
    monkeypatch,
):

    captured = {}

    class FakeClient:

        def __init__(
            self,
            **kwargs,
        ):
            captured["client"] = kwargs


    class FakeEmbeddings:

        pass


    class FakeEmbeddingProvider:

        def __init__(self):
            self.embeddings = (
                FakeEmbeddings()
            )


    class FakeVectorStore:

        @classmethod
        def from_documents(
            cls,
            **kwargs,
        ):

            captured["store"] = kwargs

            return cls()


        def similarity_search(
            self,
            query,
            k=4,
        ):
            return []


    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantClient",
        FakeClient,
    )

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.OllamaEmbeddingProvider",
        FakeEmbeddingProvider,
    )

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantVectorStore",
        FakeVectorStore,
    )

    store = QdrantStore()

    documents = [
        Document(
            page_content="Test content.",
            metadata={
                "source": "test.txt",
            },
        )
    ]

    ids = store.add_documents(
        documents
    )

    assert len(ids) == 1

    assert (
        captured["store"]["collection_name"]
        == "content_copilot"
    )

    assert (
        captured["store"]["documents"]
        == documents
    )

    assert (
        captured["store"]["embedding"]
        is store.embeddings
    )

    assert (
        captured["store"]["ids"]
        == ids
    )

def test_qdrant_store_accepts_explicit_document_ids(monkeypatch):
    """Verify caller-provided IDs are forwarded to Qdrant."""

    from langchain_core.documents import Document

    from app.retrieval.qdrant_store import QdrantStore

    captured = {}

    class FakeClient:
        """Replace the Qdrant client without making a network connection."""

        def __init__(self, **kwargs):
            captured["client"] = kwargs

    class FakeEmbeddings:
        """Stand in for the real embedding interface."""

    class FakeEmbeddingProvider:
        """Provide fake embeddings for the vector-store test."""

        def __init__(self):
            self.embeddings = FakeEmbeddings()

    class FakeVectorStore:
        """Capture documents and IDs passed to the vector store."""

        @classmethod
        def from_documents(cls, **kwargs):
            captured["store"] = kwargs
            return cls()

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantClient",
        FakeClient,
    )

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.OllamaEmbeddingProvider",
        FakeEmbeddingProvider,
    )

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantVectorStore",
        FakeVectorStore,
    )

    store = QdrantStore()

    documents = [
        Document(page_content="Stable ID test.")
    ]

    explicit_ids = [
        "stable-document-id-1",
    ]

    returned_ids = store.add_documents(
        documents,
        ids=explicit_ids,
    )

    assert returned_ids == explicit_ids
    assert captured["store"]["ids"] == explicit_ids


def test_qdrant_store_deletes_only_trailing_chunks(monkeypatch):
    """Verify cleanup filters by source ID and chunk index."""

    from app.retrieval.qdrant_store import QdrantStore

    captured = {}

    class FakeClient:
        """Capture deletion requests without contacting Qdrant."""

        def __init__(self, **kwargs):
            captured["client"] = kwargs

        def collection_exists(self, collection_name):
            captured["checked_collection"] = collection_name
            return True

        def delete(
            self,
            collection_name,
            points_selector,
            wait=True,
        ):
            captured["delete"] = {
                "collection_name": collection_name,
                "points_selector": points_selector,
                "wait": wait,
            }

    class FakeEmbeddings:
        """Stand in for the real embedding interface."""

    class FakeEmbeddingProvider:
        """Provide fake embeddings for the deletion test."""

        def __init__(self):
            self.embeddings = FakeEmbeddings()

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantClient",
        FakeClient,
    )

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.OllamaEmbeddingProvider",
        FakeEmbeddingProvider,
    )

    store = QdrantStore(
        collection_name="test_collection",
    )

    store.delete_chunks_after(
        source_id="source-123",
        last_chunk_index=2,
    )

    deletion = captured["delete"]
    conditions = deletion["points_selector"].must

    assert deletion["collection_name"] == "test_collection"
    assert deletion["wait"] is True

    assert conditions[0].key == "metadata.source_id"
    assert conditions[0].match.value == "source-123"

    assert conditions[1].key == "metadata.chunk_index"
    assert conditions[1].range.gt == 2

@pytest.mark.anyio
async def test_qdrant_store_forwards_score_threshold(monkeypatch):
    """Verify that the score threshold reaches the vector store unchanged."""

    expected_results = [
        (Document(page_content="Relevant content."), 0.62),
    ]

    vector_store = MagicMock()
    vector_store.asimilarity_search_with_score = AsyncMock(
        return_value=expected_results,
    )

    vector_store_factory = MagicMock()
    vector_store_factory.from_existing_collection.return_value = (
        vector_store
    )

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantClient",
        MagicMock(),
    )
    monkeypatch.setattr(
        "app.retrieval.qdrant_store.OllamaEmbeddingProvider",
        lambda: SimpleNamespace(embeddings=object()),
    )
    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantVectorStore",
        vector_store_factory,
    )

    store = QdrantStore(collection_name="test_collection")

    results = await store.similarity_search_with_score(
        query="A test retrieval query.",
        k=3,
        score_threshold=0.4,
    )

    assert results == expected_results

    vector_store.asimilarity_search_with_score.assert_awaited_once_with(
        "A test retrieval query.",
        k=3,
        score_threshold=0.4,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("score_threshold", [-0.01, 1.01])
async def test_qdrant_store_rejects_out_of_range_score_thresholds(
    monkeypatch,
    score_threshold,
):
    """Reject thresholds outside the application's configured range."""

    monkeypatch.setattr(
        "app.retrieval.qdrant_store.QdrantClient",
        MagicMock(),
    )
    monkeypatch.setattr(
        "app.retrieval.qdrant_store.OllamaEmbeddingProvider",
        lambda: SimpleNamespace(embeddings=object()),
    )

    store = QdrantStore(collection_name="test_collection")

    with pytest.raises(
        ValueError,
        match="score_threshold must be between 0 and 1",
    ):
        await store.similarity_search_with_score(
            query="A test retrieval query.",
            score_threshold=score_threshold,
        )