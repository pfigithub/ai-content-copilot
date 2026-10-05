from langchain_core.documents import (
    Document,
)

from app.retrieval.qdrant_store import (
    QdrantStore,
)


DEFAULT_TOP_K = 4


class ContentRetriever:

    def __init__(
        self,
        store: QdrantStore | None = None,
    ):

        self.store = (
            store
            if store is not None
            else QdrantStore()
        )

    async def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        score_threshold: float | None = None,
    ) -> list[tuple[Document, float]]:

        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        if k <= 0:
            raise ValueError(
                "k must be greater than zero."
            )

        results = await (
            self.store
            .similarity_search_with_score(
                query=query,
                k=k,
                score_threshold=score_threshold,
            )
        )

        return results