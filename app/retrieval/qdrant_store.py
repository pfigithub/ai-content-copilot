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

    def __init__(self, collection_name: str | None = None,):

        self.client = QdrantClient(
            url=settings.qdrant_url,
        )

        embedding_provider = (
            OllamaEmbeddingProvider()
        )

        self.embeddings = (
            embedding_provider.embeddings
        )
        
        self.collection_name = (
            collection_name
            if collection_name is not None
            else settings.qdrant_collection
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
                    self.collection_name
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
                        self.collection_name
                    ),
                    embedding=self.embeddings,
                    url=settings.qdrant_url,
                )
            )

        return self.store.similarity_search(
            query,
            k=k,
        )
        
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
    ):

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        if (
            score_threshold is not None
            and not 0 <= score_threshold <= 1
        ):
            raise ValueError(
                "score_threshold must be between 0 and 1."
            )

        if self.store is None:

            self.store = (
                QdrantVectorStore
                .from_existing_collection(
                    collection_name=(
                        self.collection_name
                    ),
                    embedding=self.embeddings,
                    url=settings.qdrant_url,
                )
            )

        return await (
            self.store
            .asimilarity_search_with_score(
                query,
                k=k,
                score_threshold=score_threshold,
            )
        )