from dataclasses import dataclass
from typing import (
    Annotated,
    AsyncIterator,
    Literal,
    TypedDict,
)

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.memory import InMemoryStore

from app.memory.policy import get_messages_to_remove
from app.rag.query_contextualizer import QueryContextualizer
from app.rag.rag_chain import RAGChain, RAGResponse
from app.retrieval.retriever import ContentRetriever


@dataclass
class RAGContext:
    user_id: str | None = None


class RAGGraphState(TypedDict, total=False):
    messages: Annotated[
        list[AnyMessage],
        add_messages,
    ]
    query: str
    retrieval_query: str
    k: int
    score_threshold: float | None
    results: list[tuple[Document, float]]
    response: dict
    user_preferences: list[dict]


class RAGGraph:
    def __init__(
        self,
        retriever: ContentRetriever | None = None,
        rag_chain: RAGChain | None = None,
        contextualizer: QueryContextualizer | None = None,
        checkpointer=None,
        store: InMemoryStore | None = None,
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

        self.contextualizer = (
            contextualizer
            if contextualizer is not None
            else QueryContextualizer()
        )

        self.checkpointer = (
            checkpointer
            if checkpointer is not None
            else InMemorySaver()
        )

        self.store = (
            store
            if store is not None
            else InMemoryStore()
        )

        graph = StateGraph(
            RAGGraphState,
            context_schema=RAGContext,
        )

        graph.add_node(
            "load_memory",
            self._load_memory,
        )

        graph.add_node(
            "contextualize",
            self._contextualize,
        )

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

        graph.add_node(
            "trim_memory",
            self._trim_memory,
        )

        graph.add_edge(
            START,
            "load_memory",
        )

        graph.add_edge(
            "load_memory",
            "contextualize",
        )

        graph.add_edge(
            "contextualize",
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
            "trim_memory",
        )

        graph.add_edge(
            "abstain",
            "trim_memory",
        )

        graph.add_edge(
            "trim_memory",
            END,
        )

        self.graph = graph.compile(
            checkpointer=self.checkpointer,
            store=self.store,
        )

    async def _load_memory(
        self,
        state: RAGGraphState,
        runtime: Runtime[RAGContext],
    ) -> dict:
        user_id = runtime.context.user_id

        if not user_id:
            return {
                "user_preferences": []
            }

        items = await runtime.store.asearch(
            (
                "memories",
                user_id,
            ),
            filter={
                "memory_type": "preference",
            },
            limit=10,
        )

        preferences = [
            item.value
            for item in items
        ]

        return {
            "user_preferences": preferences,
        }

    async def _contextualize(
        self,
        state: RAGGraphState,
    ) -> dict:
        messages = state.get(
            "messages",
            [],
        )

        history = messages[:-1]

        if not history:
            return {
                "retrieval_query": state["query"],
            }

        retrieval_query = (
            await self.contextualizer.contextualize(
                question=state["query"],
                history=history,
            )
        )

        return {
            "retrieval_query": retrieval_query,
        }

    async def _retrieve(
        self,
        state: RAGGraphState,
    ) -> dict:
        retrieval_query = state.get(
            "retrieval_query",
            state["query"],
        )

        results = await self.retriever.retrieve(
            query=retrieval_query,
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
        messages = state.get(
            "messages",
            [],
        )

        history = messages[:-1]

        response = (
            await self.rag_chain.answer_from_results(
                query=state["query"],
                results=state.get(
                    "results",
                    [],
                ),
                history=history,
                user_preferences=state.get(
                    "user_preferences",
                    [],
                ),
            )
        )

        return {
            "response": response.model_dump(),
            "messages": [
                AIMessage(
                    content=response.answer
                )
            ],
        }

    @staticmethod
    def _abstain(
        state: RAGGraphState,
    ) -> dict:
        response = RAGResponse(
            answer=(
                "اطلاعات مرتبط و کافی در منابع "
                "بازیابی‌شده پیدا نشد."
            ),
            sources=[],
        )

        return {
            "response": response.model_dump(),
            "messages": [
                AIMessage(
                    content=response.answer
                )
            ],
        }

    @staticmethod
    def _trim_memory(
        state: RAGGraphState,
    ) -> dict:
        messages = state.get(
            "messages",
            [],
        )

        messages_to_remove = (
            get_messages_to_remove(messages)
        )

        return {
            "messages": messages_to_remove,
        }

    async def answer(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
    ) -> RAGResponse:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        result = await self.graph.ainvoke(
            {
                "query": query,
                "k": k,
                "score_threshold": score_threshold,
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
            },
            config,
            context=RAGContext(
                user_id=user_id
            ),
        )

        return RAGResponse.model_validate(
            result["response"]
        )

    async def stream_steps(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
    ) -> AsyncIterator[dict]:
        if not query or not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        async for update in self.graph.astream(
            {
                "query": query,
                "k": k,
                "score_threshold": score_threshold,
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                    }
                ],
            },
            config,
            context=RAGContext(
                user_id=user_id
            ),
            stream_mode="updates",
        ):
            yield update

    def get_state(
        self,
        thread_id: str,
    ):
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        return self.graph.get_state(config)