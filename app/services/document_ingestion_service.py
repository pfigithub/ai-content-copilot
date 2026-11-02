import asyncio
import hashlib
from uuid import NAMESPACE_URL, uuid5

from app.models.schemas import DocumentIngestionResponse, NormalizedContent
from app.retrieval.document_splitter import (
    DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_document,
)
from app.retrieval.qdrant_store import QdrantStore

class DocumentIngestionService:
    """Coordinate content splitting, stable IDs, and vector-store updates."""

    def __init__(self, store: QdrantStore | None = None):
        self.store = store if store is not None else QdrantStore()

    @staticmethod
    def _build_source_id(content: NormalizedContent) -> str:
        """Create a stable identifier from source metadata or content hash."""
        if content.source and content.source.strip():
            source_identity = content.source.strip()
        else:
            content_hash = hashlib.sha256(content.content.strip().encode("utf-8")).hexdigest()
            source_identity = f"content-sha256:{content_hash}"
            
        # 🟢 FIX: Include user_id in the deterministic UUID to prevent cross-tenant collision
        safe_user = content.user_id if content.user_id else "global_tenant"
        identity = f"ai-content-copilot:{safe_user}:{content.source_type}:{source_identity}"
        return str(uuid5(NAMESPACE_URL, identity))

    async def ingest(
        self,
        content: NormalizedContent,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> DocumentIngestionResponse:
        # بقیه کدهای این متد دقیقاً مثل قبل باقی می‌ماند...
        if not content.content or not content.content.strip():
            raise ValueError("Content cannot be empty.")

        documents = await asyncio.to_thread(
            split_document,
            text=content.content,
            source=content.source,
            source_type=content.source_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            pages=content.pages,
        )

        source_id = self._build_source_id(content)
        for document in documents:
            document.metadata["source_id"] = source_id
            if content.user_id:
                document.metadata["user_id"] = content.user_id

        document_ids = [
            str(uuid5(NAMESPACE_URL, f"{source_id}:chunk:{document.metadata['chunk_index']}"))
            for document in documents
        ]

        stored_ids = await asyncio.to_thread(self.store.add_documents, documents, document_ids)
        if stored_ids != document_ids:
            raise RuntimeError("The vector store returned IDs that do not match the requested document IDs.")

        await asyncio.to_thread(self.store.delete_chunks_after, source_id, len(documents))

        return DocumentIngestionResponse(
            source_type=content.source_type,
            source=content.source,
            chunks_created=len(documents),
            document_ids=stored_ids,
        )