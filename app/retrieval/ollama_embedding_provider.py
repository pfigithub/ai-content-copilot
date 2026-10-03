from langchain_ollama import (
    OllamaEmbeddings,
)

from app.config import settings
from app.retrieval.embedding_provider import (
    EmbeddingProvider,
)


class OllamaEmbeddingProvider(
    EmbeddingProvider,
):

    def __init__(
        self,
    ):

        self.embeddings = OllamaEmbeddings(
            model=settings.ollama_embedding_model,
            base_url=settings.ollama_base_url,
        )

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        if any(
            not text or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Document text cannot be empty."
            )

        return await (
            self.embeddings.aembed_documents(
                texts
            )
        )

    async def embed_query(
        self,
        text: str,
    ) -> list[float]:

        if not text or not text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        return await (
            self.embeddings.aembed_query(
                text
            )
        )