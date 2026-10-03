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