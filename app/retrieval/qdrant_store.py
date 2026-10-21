from uuid import uuid4

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from app.config import settings
from app.retrieval.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)


class QdrantStore:
    """Manage document insertion, retrieval, and source-specific cleanup in Qdrant."""

    def __init__(
        self,
        collection_name: str | None = None,
    ):
        """Initialize the Qdrant client and embedding provider."""

        self.client = QdrantClient(
            url=settings.qdrant_url,
        )

        embedding_provider = OllamaEmbeddingProvider()
        self.embeddings = embedding_provider.embeddings

        self.collection_name = (
            collection_name
            if collection_name is not None
            else settings.qdrant_collection
        )

        self.store: QdrantVectorStore | None = None

    def _get_store(self) -> QdrantVectorStore:
        """Return the vector store, connecting to an existing collection when needed."""

        if self.store is None:
            self.store = QdrantVectorStore.from_existing_collection(
                collection_name=self.collection_name,
                embedding=self.embeddings,
                url=settings.qdrant_url,
            )

        return self.store

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:
        """Insert documents using supplied IDs or generate new random IDs."""

        if not documents:
            if ids:
                raise ValueError(
                    "IDs cannot be supplied when documents are empty."
                )

            return []

        if ids is None:
            document_ids = [
                str(uuid4())
                for _ in documents
            ]
        else:
            if len(ids) != len(documents):
                raise ValueError(
                    "The number of IDs must match the number of documents."
                )

            if any(not document_id.strip() for document_id in ids):
                raise ValueError(
                    "Document IDs cannot be empty."
                )

            if len(set(ids)) != len(ids):
                raise ValueError(
                    "Document IDs must be unique within a batch."
                )

            document_ids = list(ids)

        # Explicit IDs allow repeated ingestion to update existing points
        # instead of creating new points for the same source and chunk.
        self.store = QdrantVectorStore.from_documents(
            documents=documents,
            embedding=self.embeddings,
            url=settings.qdrant_url,
            collection_name=self.collection_name,
            ids=document_ids,
        )

        return document_ids

    def delete_chunks_after(
        self,
        source_id: str,
        last_chunk_index: int,
    ) -> None:
        """Delete obsolete trailing chunks after a source is re-ingested."""

        if not source_id or not source_id.strip():
            raise ValueError(
                "Source ID cannot be empty."
            )

        if last_chunk_index < 0:
            raise ValueError(
                "Last chunk index cannot be negative."
            )

        if not self.client.collection_exists(
            collection_name=self.collection_name,
        ):
            return

        # LangChain stores document metadata under the Qdrant "metadata"
        # payload key by default, so these filters target that nested data.
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_id",
                        match=MatchValue(
                            value=source_id,
                        ),
                    ),
                    FieldCondition(
                        key="metadata.chunk_index",
                        range=Range(
                            gt=last_chunk_index,
                        ),
                    ),
                ],
            ),
            wait=True,
        )

    def similarity_search(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Return the documents most similar to a query."""

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        return self._get_store().similarity_search(
            query,
            k=k,
        )

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:
        """Return relevant documents together with their retrieval scores."""

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

        return await self._get_store().asimilarity_search_with_score(
            query,
            k=k,
            score_threshold=score_threshold,
        )