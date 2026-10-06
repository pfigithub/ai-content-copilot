from typing import AsyncIterator, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.rag.rag_chain import RAGChain, RAGResponse
from app.retrieval.retriever import ContentRetriever


class RAGGraphState(TypedDict, total=False):
    query: str
    k: int
    score_threshold: float | None
    results: list[tuple[Document, float]]
    response: RAGResponse


class RAGGraph:
    def __init__(
        self,
        retriever: ContentRetriever | None = None,
        rag_chain: RAGChain | None = None,
    ):
        self.retriever = (
            retriever
            if retriever is not None
            else ContentRetriever()
        )

        self.rag_chain = (
            rag_chain
            if rag_chain is not None
            else RAGChain(
                retriever=self.retriever
            )
        )

        graph = StateGraph(RAGGraphState)

        graph.add_node(
            "retrieve",
            self._retrieve,
        )

        graph.add_node(
            "generate",
            self._generate,
        )

        graph.add_node(
            "abstain",
            self._abstain,
        )

        graph.add_edge(
            START,
            "retrieve",
        )

        graph.add_conditional_edges(
            "retrieve",
            self._route_after_retrieve,
            {
                "generate": "generate",
                "abstain": "abstain",
            },
        )

        graph.add_edge(
            "generate",
            END,
        )

        graph.add_edge(
            "abstain",
            END,
        )

        self.graph = graph.compile()

    async def _retrieve(
        self,
        state: RAGGraphState,
    ) -> dict:
        results = await self.retriever.retrieve(
            query=state["query"],
            k=state.get("k", 4),
            score_threshold=state.get(
                "score_threshold"
            ),
        )

        return {
            "results": results,
        }

    @staticmethod
    def _route_after_retrieve(
        state: RAGGraphState,
    ) -> Literal["generate", "abstain"]:
        if state.get("results"):
            return "generate"

        return "abstain"

    async def _generate(
        self,
        state: RAGGraphState,
    ) -> dict:
        response = await self.rag_chain.answer_from_results(
            query=state["query"],
            results=state.get("results", []),
        )

        return {
            "response": response,
        }

    @staticmethod
    def _abstain(
        state: RAGGraphState,
    ) -> dict:
        return {
            "response": RAGResponse(
                answer=(
                    "اطلاعات مرتبط و کافی در منابع "
                    "بازیابی‌شده پیدا نشد."
                ),
                sources=[],
            )
        }
        
    async def stream_steps(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
    ) -> AsyncIterator[dict]:
        if not query or not query.strip():
            raise ValueError(
            "   Query cannot be empty."
            )

        async for update in self.graph.astream(
            {
                "query": query,
                "k": k,
                "score_threshold": score_threshold,
            },
            stream_mode="updates",
        ):
            yield update

    
    async def answer(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
    ) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        result = await self.graph.ainvoke(
            {
                "query": query,
                "k": k,
                "score_threshold": score_threshold,
            }
        )

        return result["response"]