
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
from app.retrieval.ollama_embedding_provider import OllamaEmbeddingProvider


class QdrantStore:
    """Provide tenant-scoped document storage and retrieval in Qdrant."""

    def __init__(self, collection_name: str | None = None):
        """Initialize the Qdrant client and embedding provider."""

        self.client = QdrantClient(url=settings.qdrant_url)

        embedding_provider = OllamaEmbeddingProvider()
        self.embeddings = embedding_provider.embeddings

        self.collection_name = (
            collection_name
            if collection_name is not None
            else settings.qdrant_collection
        )

        self.store: QdrantVectorStore | None = None

    @staticmethod
    def _validate_user_id(user_id: str | None) -> str:
        """Reject missing or malformed tenant identifiers."""

        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(
                "Security Error: a valid user_id is required."
            )

        return user_id.strip()

    def _get_store(self) -> QdrantVectorStore:
        """Create the vector-store adapter lazily."""

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
        *,
        user_id: str,
    ) -> list[str]:
        """Store documents only when every document belongs to the tenant."""

        tenant_id = self._validate_user_id(user_id)

        if not documents:
            if ids:
                raise ValueError(
                    "IDs cannot be supplied when documents are empty."
                )
            return []

        for index, document in enumerate(documents):
            metadata = document.metadata

            if metadata.get("user_id") != tenant_id:
                raise ValueError(
                    f"Document at index {index} does not belong to "
                    "the authenticated tenant."
                )

            source_id = metadata.get("source_id")
            if not isinstance(source_id, str) or not source_id.strip():
                raise ValueError(
                    f"Document at index {index} is missing source_id metadata."
                )

            chunk_index = metadata.get("chunk_index")
            if not isinstance(chunk_index, int) or chunk_index < 1:
                raise ValueError(
                    f"Document at index {index} has an invalid chunk_index."
                )

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

            if any(
                not isinstance(document_id, str) or not document_id.strip()
                for document_id in ids
            ):
                raise ValueError(
                    "Document IDs cannot be empty."
                )

            if len(set(ids)) != len(ids):
                raise ValueError(
                    "Document IDs must be unique within a batch."
                )

            document_ids = list(ids)

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
        *,
        user_id: str,
    ) -> None:
        """Delete obsolete chunks only within the owning tenant."""

        tenant_id = self._validate_user_id(user_id)

        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("Source ID cannot be empty.")

        if last_chunk_index < 0:
            raise ValueError(
                "Last chunk index cannot be negative."
            )

        if not self.client.collection_exists(
            collection_name=self.collection_name
        ):
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.user_id",
                        match=MatchValue(value=tenant_id),
                    ),
                    FieldCondition(
                        key="metadata.source_id",
                        match=MatchValue(value=source_id),
                    ),
                    FieldCondition(
                        key="metadata.chunk_index",
                        range=Range(gt=last_chunk_index),
                    ),
                ],
            ),
            wait=True,
        )

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
        user_id: str | None = None,
    ) -> list[tuple[Document, float]]:
        """Retrieve only documents owned by the authenticated tenant."""

        tenant_id = self._validate_user_id(user_id)

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

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

        search_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.user_id",
                    match=MatchValue(value=tenant_id),
                )
            ]
        )

        return await self._get_store().asimilarity_search_with_score(
            query,
            k=k,
            score_threshold=score_threshold,
            filter=search_filter,
        )
