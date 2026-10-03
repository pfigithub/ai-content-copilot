from uuid import uuid4

from langchain_core.documents import (
    Document,
)
from langchain_qdrant import (
    QdrantVectorStore,
)
from qdrant_client import (
    QdrantClient,
)

from app.config import settings
from app.retrieval.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)


class QdrantStore:

    def __init__(self):

        self.client = QdrantClient(
            url=settings.qdrant_url,
        )

        embedding_provider = (
            OllamaEmbeddingProvider()
        )

        self.embeddings = (
            embedding_provider.embeddings
        )

        self.store = None

    def add_documents(
        self,
        documents: list[Document],
    ) -> list[str]:

        if not documents:
            return []

        ids = [
            str(uuid4())
            for _ in documents
        ]

        self.store = (
            QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embeddings,
                url=settings.qdrant_url,
                collection_name=(
                    settings.qdrant_collection
                ),
                ids=ids,
            )
        )

        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ):

        if self.store is None:

            self.store = (
                QdrantVectorStore
                .from_existing_collection(
                    collection_name=(
                        settings.qdrant_collection
                    ),
                    embedding=self.embeddings,
                    url=settings.qdrant_url,
                )
            )

        return self.store.similarity_search(
            query,
            k=k,
        )