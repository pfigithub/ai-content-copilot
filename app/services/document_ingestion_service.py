
import asyncio
import hashlib
from uuid import NAMESPACE_URL, uuid5

from app.models.schemas import (
    DocumentIngestionResponse,
    NormalizedContent,
)
from app.retrieval.document_splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_document,
)
from app.retrieval.qdrant_store import QdrantStore


class DocumentIngestionService:
    """Coordinate tenant-safe document splitting and vector-store updates."""

    def __init__(self, store: QdrantStore | None = None):
        """Use the supplied vector store or create the default store."""

        self.store = (
            store
            if store is not None
            else QdrantStore()
        )

    @staticmethod
    def _validate_user_id(user_id: str | None) -> str:
        """Require a server-assigned owner before document ingestion."""

        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(
                "Security Error: user_id is required for document ingestion."
            )

        return user_id.strip()

    @staticmethod
    def _build_source_id(
        content: NormalizedContent,
        user_id: str,
    ) -> str:
        """Create a deterministic source ID scoped to one tenant."""

        if content.source and content.source.strip():
            source_identity = content.source.strip()
        else:
            content_hash = hashlib.sha256(
                content.content.strip().encode("utf-8")
            ).hexdigest()

            source_identity = f"content-sha256:{content_hash}"

        identity = (
            f"ai-content-copilot:"
            f"{user_id}:"
            f"{content.source_type}:"
            f"{source_identity}"
        )

        return str(
            uuid5(NAMESPACE_URL, identity)
        )

    async def ingest(
        self,
        content: NormalizedContent,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> DocumentIngestionResponse:
        """Split, validate, store, and safely replace a tenant's source."""

        if not content.content or not content.content.strip():
            raise ValueError(
                "Content cannot be empty."
            )

        # Never fall back to a shared or anonymous tenant.
        user_id = self._validate_user_id(content.user_id)

        documents = await asyncio.to_thread(
            split_document,
            text=content.content,
            source=content.source,
            source_type=content.source_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pages=content.pages,
        )

        source_id = self._build_source_id(
            content=content,
            user_id=user_id,
        )

        for document in documents:
            document.metadata["source_id"] = source_id
            document.metadata["user_id"] = user_id

        document_ids = [
            str(
                uuid5(
                    NAMESPACE_URL,
                    f"{source_id}:chunk:{document.metadata['chunk_index']}",
                )
            )
            for document in documents
        ]

        stored_ids = await asyncio.to_thread(
            self.store.add_documents,
            documents,
            document_ids,
            user_id=user_id,
        )

        if stored_ids != document_ids:
            raise RuntimeError(
                "The vector store returned IDs that do not match "
                "the requested document IDs."
            )

        # Remove stale chunks only after the new chunks have been stored.
        await asyncio.to_thread(
            self.store.delete_chunks_after,
            source_id,
            len(documents),
            user_id=user_id,
        )

        return DocumentIngestionResponse(
            source_type=content.source_type,
            source=content.source,
            chunks_created=len(documents),
            document_ids=stored_ids,
        )
