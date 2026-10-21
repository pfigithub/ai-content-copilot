from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
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
                        "CORE EVIDENCE RULES:\n"
                        "1. Answer the user's question using only the provided context.\n"
                        "2. Conversation history is contextual information, not evidence.\n"
                        "3. User preferences are for personalization only, not factual evidence.\n"
                        "4. Treat previous assistant messages as unverified. Do not use them "
                        "as factual evidence.\n"
                        "5. Do not add facts, measurements, events, or source details that "
                        "are not supported by the context.\n"
                        "6. Answer in the same language as the user's question, unless "
                        "the user explicitly requests another language.\n\n"
                        "DOCUMENT RELATIONSHIPS AND HYPOTHETICAL SOURCES:\n"
                        "SOURCE ATTRIBUTION:\n"
                        "Do not claim that one document implements, executes, validates, "
                        "or is based on another document unless the source text explicitly "
                        "establishes that relationship.\n"
                        "Do not describe a proposed, hypothetical, or synthetic workflow as "
                        "something that actually happened, was deployed, or was executed in "
                        "the real world.\n"
                        "If a source explicitly labels its content as synthetic or test-only, "
                        "preserve that distinction consistently throughout the entire answer. "
                        "Do not imply real-world execution elsewhere in the answer.\n"
                        "Describe thematic similarities as similarities, not proof of an "
                        "implementation relationship.\n"
                        "Do not infer that a capability was actually implemented merely because "
                        "a document describes that capability or includes it in a proposed workflow.\n\n"
                        "When making factual claims about the documents, identify the "
                        "supporting source using its exact source name from the context. "
                        "Distinguish what a document explicitly states from your own "
                        "comparison or interpretation of the provided documents. Never "
                        "invent a citation or attribute a claim to an unrelated source.\n\n"
                        "EVIDENCE SUFFICIENCY:\n"
                        "Evaluate each part of the question separately. If the context does "
                        "not establish a requested conclusion, clearly say that the provided "
                        "Do not guess or present a possibility as a verified result.\n"
                        "or present a possibility as a verified result.\n"
                        "If the user asks whether costs, error rates, performance improvements, "
                        "or real-world success can be concluded, answer based on whether the "
                        "provided context actually contains supporting measurements or "
                        "outcome evidence. If it does not, explicitly state that such a "
                        "conclusion cannot be drawn from these sources. Do not say that it "
                        "can be concluded and then contradict yourself by saying evidence "
                        "is missing.\n"
                        "The absence of evidence for a result does not prove that the result "
                        "was unsuccessful;state only what the provided sources establish.\n\n"
                        "COMPARATIVE QUESTIONS:\n"
                        "When comparing multiple documents, explain their supported "
                        "similarities and differences, attribute each document-specific "
                        "claim to its source, and clearly identify any unanswered part "
                        "of the question. Do not assume that a hypothetical or experimental "
                        "description proves real-world effectiveness.\n\n"
                        "FINAL CONSISTENCY CHECK:\n"
                        "Before producing the answer, check that its conclusions do not "
                        "contradict one another. In particular, never claim that the sources "
                        "establish a fact if you also state that the evidence needed to "
                        "establish that fact is missing. Resolve any contradiction before "
                        "returning the final answer.\n"
                        "If the context is insufficient, say so clearly instead of guessing."
                    ),
                ),
                (
                    "human",
                    (
                        "User preferences:\n"
                        "{user_preferences}\n\n"
                        "Conversation history:\n"
                        "{history}\n\n"
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

        return await self.answer_from_results(
            query=query,
            results=results,
        )

    async def answer_from_results(
        self,
        query: str,
        results: list[tuple[Document, float]],
        history: list[BaseMessage] | None = None,
        user_preferences: list[dict] | None = None,
    ) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

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
        
        history_text = self._format_history(
            history or []
        )
        
        user_preferences_text = self._format_preferences(
            user_preferences or []
        )

        result = await self.chain.ainvoke(
            {
                "user_preferences": user_preferences_text,
                "history": history_text,
                "question": query,
                "context": context,
            }
        )

        return RAGResponse(
            answer=result.answer,
            sources=sources,
        )

    @staticmethod
    def _format_history(
        messages: list[BaseMessage],
    ) -> str:
        if not messages :
            return "No previous conversation."
        
        parts = []
        
        for message in messages:
            if message.type == "human":
                role = "User"
            elif message.type == "ai":
                role = "Assistant"
            else:
                role = message.type
                
            parts.append(
                f"{role}: {message.content}"
            )
    
        return "\n".join(parts)
    
    @staticmethod
    def _format_preferences(
        preferences: list[dict],
    ) -> str:
        if not preferences:
            return "No user preferences."

        parts = []

        for preference in preferences:
            content = str(
                preference.get("content", "")
            ).strip()

            if content:
                parts.append(
                    f"- {content}"
                )

        return (
            "\n".join(parts)
            if parts
            else "No user preferences."
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