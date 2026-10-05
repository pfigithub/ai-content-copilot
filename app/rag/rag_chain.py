from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama

from app.config import settings
from app.retrieval.retriever import ContentRetriever


class RAGAnswer(BaseModel):
    answer: str


class RAGSource(BaseModel):
    source: str | None = None
    source_type: str | None = None
    chunk_index: int | None = None
    chunk_count: int | None = None
    score: float


class RAGResponse(BaseModel):
    answer: str
    sources: list[RAGSource]


class RAGChain:
    def __init__(
        self,
        retriever: ContentRetriever | None = None,
        model: Runnable | None = None,
    ):
        self.retriever = (
            retriever
            if retriever is not None
            else ContentRetriever()
        )

        if model is None:
            model = ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a grounded research assistant.\n\n"
                        "Answer the user's question using only the provided context.\n"
                        "Do not add facts that are not supported by the context.\n"
                        "If the context does not contain enough information "
                        "to answer the question, say so clearly instead of guessing.\n"
                    ),
                ),
                (
                    "human",
                    (
                        "Question:\n"
                        "{question}\n\n"
                        "Context:\n"
                        "{context}"
                    ),
                ),
            ]
        )

        structured_model = model.with_structured_output(
            RAGAnswer
        )

        self.chain = prompt | structured_model

    async def answer(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
    ) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        results = await self.retriever.retrieve(
            query=query,
            k=k,
            score_threshold=score_threshold,
        )

        if not results:
            return RAGResponse(
                answer=(
                    "اطلاعات مرتبط و کافی در منابع "
                    "بازیابی‌شده پیدا نشد."
                ),
                sources=[],
            )

        context_parts = []
        sources = []

        for index, (document, score) in enumerate(
            results,
            start=1,
        ):
            context_parts.append(
                self._format_document(
                    index=index,
                    document=document,
                )
            )

            sources.append(
                RAGSource(
                    source=document.metadata.get("source"),
                    source_type=document.metadata.get(
                        "source_type"
                    ),
                    chunk_index=document.metadata.get(
                        "chunk_index"
                    ),
                    chunk_count=document.metadata.get(
                        "chunk_count"
                    ),
                    score=score,
                )
            )

        context = "\n\n".join(context_parts)

        result = await self.chain.ainvoke(
            {
                "question": query,
                "context": context,
            }
        )

        return RAGResponse(
            answer=result.answer,
            sources=sources,
        )

    @staticmethod
    def _format_document(
        index: int,
        document: Document,
    ) -> str:
        source = (
            document.metadata.get("source")
            or "unknown"
        )

        source_type = (
            document.metadata.get("source_type")
            or "unknown"
        )

        chunk_index = document.metadata.get(
            "chunk_index"
        )

        chunk_count = document.metadata.get(
            "chunk_count"
        )

        return (
            f"[SOURCE {index}]\n"
            f"source: {source}\n"
            f"source_type: {source_type}\n"
            f"chunk: {chunk_index}/{chunk_count}\n"
            f"content:\n"
            f"{document.page_content}"
        )