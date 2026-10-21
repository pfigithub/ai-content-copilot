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

from app.retrieval.qdrant_store import (
    QdrantStore,
)


class DocumentIngestionService:
    """Coordinate content splitting, stable IDs, and vector-store updates."""

    def __init__(
        self,
        store: QdrantStore | None = None,
    ):
        """Initialize the service with an injectable vector store."""

        self.store = (
            store
            if store is not None
            else QdrantStore()
        )

    @staticmethod
    def _build_source_id(
        content: NormalizedContent,
    ) -> str:
        """Create a stable identifier from source metadata or content hash."""

        if content.source and content.source.strip():
            source_identity = content.source.strip()
        else:
            # Identical unsourced content receives the same identity.
            content_hash = hashlib.sha256(
                content.content.strip().encode("utf-8")
            ).hexdigest()

            source_identity = f"content-sha256:{content_hash}"

        identity = (
            f"ai-content-copilot:"
            f"{content.source_type}:"
            f"{source_identity}"
        )

        return str(
            uuid5(
                NAMESPACE_URL,
                identity,
            )
        )

    async def ingest(
        self,
        content: NormalizedContent,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> DocumentIngestionResponse:
        """Split content and insert or update its chunks in Qdrant."""

        if not content.content or not content.content.strip():
            raise ValueError(
                "Content cannot be empty."
            )

        documents = await asyncio.to_thread(
            split_document,
            text=content.content,
            source=content.source,
            source_type=content.source_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        source_id = self._build_source_id(content)

        for document in documents:
            # Every chunk from the same logical source shares one stable ID.
            document.metadata["source_id"] = source_id

        # Stable chunk IDs update existing points when the same source
        # is ingested again. Chunk indexes start at one in split_document.
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
        )

        if stored_ids != document_ids:
            raise RuntimeError(
                "The vector store returned IDs that do not match "
                "the requested document IDs."
            )

        # Remove old trailing chunks when an updated source becomes shorter.
        # This runs only after the new batch has been stored successfully.
        await asyncio.to_thread(
            self.store.delete_chunks_after,
            source_id,
            len(documents),
        )

        return DocumentIngestionResponse(
            source_type=content.source_type,
            source=content.source,
            chunks_created=len(documents),
            document_ids=stored_ids,
        )