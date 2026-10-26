import logging
from dataclasses import dataclass
from typing import (
    Annotated,
    AsyncIterator,
    Literal,
    TypedDict,
)

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.runtime import Runtime
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from app.memory.policy import get_messages_to_remove
from app.rag.research_query_planner import ResearchQueryPlanner
from app.rag.evidence_sufficiency import (
    EvidenceSufficiencyEvaluator,
)
from app.rag.query_contextualizer import QueryContextualizer
from app.rag.rag_chain import RAGChain, RAGResponse
from app.retrieval.retriever import ContentRetriever
from app.services.guardrail_service import InputGuardrail, OutputGuardrail, SecurityViolationError


logger = logging.getLogger(__name__)

# Initial candidate based on the current small retrieval evaluation.
DEFAULT_SCORE_THRESHOLD = 0.4
MAX_ALTERNATIVE_ATTEMPTS = 1


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
    research_attempt: int
    research_plan_failed: bool
    evidence_sufficient: bool
    evidence_assessment_reason: str


class RAGGraph:
    def __init__(
        self,
        retriever: ContentRetriever | None = None,
        rag_chain: RAGChain | None = None,
        contextualizer: QueryContextualizer | None = None,
        checkpointer=None,
        store: BaseStore | None = None,
        query_planner: ResearchQueryPlanner | None = None,
        evidence_evaluator: EvidenceSufficiencyEvaluator | None = None,
        output_guardrail: OutputGuardrail | None = None,
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

        self.query_planner = query_planner
        self.evidence_evaluator = evidence_evaluator
        self.output_guardrail = output_guardrail if output_guardrail is not None else OutputGuardrail()

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
            "evaluate_evidence",
            self._evaluate_evidence,
        )

        graph.add_node(
            "plan_alternative_query",
            self._plan_alternative_query,
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
                "evaluate_evidence": "evaluate_evidence",
                "plan_alternative_query": "plan_alternative_query",
                "abstain": "abstain",
            },
        )

        graph.add_conditional_edges(
            "evaluate_evidence",
            self._route_after_evaluation,
            {
                "generate": "generate",
                "plan_alternative_query": "plan_alternative_query",
                "abstain": "abstain",
            },
        )

        graph.add_conditional_edges(
            "plan_alternative_query",
            self._route_after_plan,
            {
                "retrieve": "retrieve",
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

        return {
            "user_preferences": [
                item.value
                for item in items
            ],
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
        """Retrieve evidence and discard results below the configured threshold."""

        retrieval_query = state.get(
            "retrieval_query",
            state["query"],
        )

        score_threshold = state.get(
            "score_threshold",
            DEFAULT_SCORE_THRESHOLD,
        )

        results = await self.retriever.retrieve(
            query=retrieval_query,
            k=state.get("k", 4),
            score_threshold=score_threshold,
        )

        if score_threshold is not None:
            results = [
                (document, score)
                for document, score in results
                if score >= score_threshold
            ]

        return {
            "results": results,
        }

    async def _evaluate_evidence(
        self,
        state: RAGGraphState,
    ) -> dict:
        """Evaluate retrieved evidence before deciding whether to answer."""

        if self.evidence_evaluator is None:
            return {
                "evidence_sufficient": True,
                "evidence_assessment_reason": (
                    "Evaluator not configured; using the legacy path."
                ),
            }

        try:
            assessment = await self.evidence_evaluator.evaluate(
                question=state["query"],
                results=state.get("results", []),
            )
        except Exception:
            logger.warning(
                "Evidence evaluation failed; treating evidence as insufficient.",
                exc_info=True,
            )
            return {
                "evidence_sufficient": False,
                "evidence_assessment_reason": (
                    "Evaluation failed; an alternative retrieval is required."
                ),
            }

        return {
            "evidence_sufficient": assessment.sufficient,
            "evidence_assessment_reason": assessment.reason,
        }


    @staticmethod
    def _route_after_evaluation(
        state: RAGGraphState,
    ) -> Literal["generate", "plan_alternative_query", "abstain"]:
        """Route based on evidence quality and the bounded retry budget."""

        if state.get("evidence_sufficient", False):
            return "generate"

        if state.get("research_attempt", 0) < MAX_ALTERNATIVE_ATTEMPTS:
            return "plan_alternative_query"

        return "abstain"


    def _route_after_retrieve(
        self,
        state: RAGGraphState,
    ) -> Literal[
        "generate",
        "evaluate_evidence",
        "plan_alternative_query",
        "abstain",
    ]:
        """Route retrieval results to evaluation, generation, or a bounded retry."""

        if state.get("results"):
            if self.evidence_evaluator is not None:
                return "evaluate_evidence"
            return "generate"

        if state.get("research_attempt", 0) < MAX_ALTERNATIVE_ATTEMPTS:
            return "plan_alternative_query"

        return "abstain"



    async def _plan_alternative_query(
        self,
        state: RAGGraphState,
    ) -> dict:
        """Plan one different retrieval query after the initial search fails."""

        if self.query_planner is None:
            return {
                "research_attempt": MAX_ALTERNATIVE_ATTEMPTS,
                "research_plan_failed": True,
            }

        attempted_query = state.get(
            "retrieval_query",
            state["query"],
        )

        try:
            alternative_query = (
                await self.query_planner.create_alternative_query(
                    question=state["query"],
                    attempted_query=attempted_query,
                )
            )
        except Exception:
            logger.warning(
                "Alternative retrieval planning failed; abstaining.",
                exc_info=True,
            )
            alternative_query = None

        candidate = (alternative_query or "").strip()

        if (
            not candidate
            or candidate.casefold() == attempted_query.strip().casefold()
        ):
            return {
                "research_attempt": MAX_ALTERNATIVE_ATTEMPTS,
                "research_plan_failed": True,
            }

        return {
            "retrieval_query": candidate,
            "research_attempt": MAX_ALTERNATIVE_ATTEMPTS,
            "research_plan_failed": False,
            "k": min(max(state.get("k", 4) * 2, 8), 10),
        }


    @staticmethod
    def _route_after_plan(
        state: RAGGraphState,
    ) -> Literal["retrieve", "abstain"]:
        """Continue only when a distinct alternative query was produced."""

        if state.get("research_plan_failed", True):
            return "abstain"

        return "retrieve"

    async def _generate(
        self,
        state: RAGGraphState,
    ) -> dict:
        """Generate a grounded answer from the retrieved evidence."""

        messages = state.get("messages", [])
        history = messages[:-1]

        response = await self.rag_chain.answer_from_results(
            query=state["query"],
            results=state.get("results", []),
            history=history,
            user_preferences=state.get("user_preferences", []),
        )

        # Execute output guardrail to prevent data leakage or harmful content
        try:
            if self.output_guardrail is not None:
                await self.output_guardrail.verify_safe_output(response.answer)
        except SecurityViolationError as exc:
            logger.warning("Output guardrail triggered", exc_info=True)
            # Replace malicious response with a safe fallback message
            response = RAGResponse(
                answer="The response was blocked due to security policy violations.",
                sources=[],
            )

        return {
            "response": response.model_dump(),
            "messages": [AIMessage(content=response.answer)],
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

        return {
            "messages": get_messages_to_remove(
                messages
            ),
        }

    async def answer(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = DEFAULT_SCORE_THRESHOLD,
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
                "research_attempt": 0,
                "research_plan_failed": False,
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
        score_threshold: float | None = DEFAULT_SCORE_THRESHOLD,
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
                "research_attempt": 0,
                "research_plan_failed": False,
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

    async def aget_state(
        self,
        thread_id: str,
    ):
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }

        return await self.graph.aget_state(config)