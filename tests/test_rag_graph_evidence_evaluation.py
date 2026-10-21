"""Test evidence assessment and bounded research routing."""

import asyncio

from langchain_core.documents import Document

from app.rag.evidence_sufficiency import EvidenceAssessment
from app.rag.rag_chain import RAGResponse, RAGSource
from app.rag.rag_graph import RAGGraph


class FakeRetriever:
    """Return predefined evidence and record every retrieval request."""

    def __init__(self, result_sets):
        self.result_sets = result_sets
        self.calls = []

    async def retrieve(self, query, k=4, score_threshold=None):
        self.calls.append(
            {
                "query": query,
                "k": k,
                "score_threshold": score_threshold,
            }
        )
        index = len(self.calls) - 1
        if index >= len(self.result_sets):
            return []
        return self.result_sets[index]


class FakeEvidenceEvaluator:
    """Return predefined assessments without invoking a language model."""

    def __init__(self, assessments):
        self.assessments = assessments
        self.calls = []

    async def evaluate(self, question, results):
        self.calls.append(
            {
                "question": question,
                "sources": [
                    document.metadata.get("source", "unknown")
                    for document, _score in results
                ],
            }
        )
        outcome = self.assessments[len(self.calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeQueryPlanner:
    """Provide a predictable alternative query for graph tests."""

    def __init__(self, alternative_query="alternative retrieval query"):
        self.alternative_query = alternative_query
        self.calls = []

    async def create_alternative_query(self, question, attempted_query):
        self.calls.append(
            {
                "question": question,
                "attempted_query": attempted_query,
            }
        )
        return self.alternative_query


class FakeAnswerChain:
    """Record generation calls and return a predictable RAG response."""

    def __init__(self):
        self.calls = []

    async def answer_from_results(
        self,
        query,
        results,
        history=None,
        user_preferences=None,
    ):
        self.calls.append(
            {
                "query": query,
                "sources": [
                    document.metadata.get("source", "unknown")
                    for document, _score in results
                ],
            }
        )
        sources = [
            RAGSource(
                source=document.metadata.get("source"),
                source_type=document.metadata.get("source_type"),
                chunk_index=document.metadata.get("chunk_index"),
                chunk_count=document.metadata.get("chunk_count"),
                score=score,
            )
            for document, score in results
        ]
        return RAGResponse(
            answer="Grounded test answer.",
            sources=sources,
        )


def make_results(source, content):
    """Create one predictable source chunk for retrieval tests."""

    return [
        (
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "source_type": "text",
                    "chunk_index": 1,
                    "chunk_count": 1,
                },
            ),
            0.8,
        )
    ]


def test_sufficient_evidence_generates_without_retry():
    retriever = FakeRetriever(
        result_sets=[
            make_results("first.txt", "Evidence that answers the question."),
        ]
    )
    evaluator = FakeEvidenceEvaluator(
        [
            EvidenceAssessment(
                sufficient=True,
                reason="The source supports the requested answer.",
            )
        ]
    )
    planner = FakeQueryPlanner()
    answer_chain = FakeAnswerChain()
    graph = RAGGraph(
        retriever=retriever,
        rag_chain=answer_chain,
        query_planner=planner,
        evidence_evaluator=evaluator,
    )

    response = asyncio.run(
        graph.answer(
            query="What does the source establish?",
            thread_id="evidence-sufficient-test",
        )
    )

    assert response.answer == "Grounded test answer."
    assert len(retriever.calls) == 1
    assert len(evaluator.calls) == 1
    assert planner.calls == []
    assert len(answer_chain.calls) == 1


def test_insufficient_evidence_retries_and_generates_when_sufficient():
    retriever = FakeRetriever(
        result_sets=[
            make_results("initial.txt", "Related but incomplete evidence."),
            make_results("alternative.txt", "Evidence that covers the question."),
        ]
    )
    evaluator = FakeEvidenceEvaluator(
        [
            EvidenceAssessment(
                sufficient=False,
                reason="The first source does not cover the full question.",
            ),
            EvidenceAssessment(
                sufficient=True,
                reason="The second retrieval provides sufficient evidence.",
            ),
        ]
    )
    planner = FakeQueryPlanner("alternative retrieval query")
    answer_chain = FakeAnswerChain()
    graph = RAGGraph(
        retriever=retriever,
        rag_chain=answer_chain,
        query_planner=planner,
        evidence_evaluator=evaluator,
    )

    response = asyncio.run(
        graph.answer(
            query="Compare the evidence and identify its limitations.",
            k=4,
            thread_id="evidence-retry-success-test",
        )
    )

    assert response.answer == "Grounded test answer."
    assert len(retriever.calls) == 2
    assert [call["k"] for call in retriever.calls] == [4, 8]
    assert retriever.calls[1]["query"] == "alternative retrieval query"
    assert len(evaluator.calls) == 2
    assert len(planner.calls) == 1
    assert len(answer_chain.calls) == 1
    assert answer_chain.calls[0]["sources"] == ["alternative.txt"]


def test_insufficient_evidence_after_retry_abstains():
    retriever = FakeRetriever(
        result_sets=[
            make_results("initial.txt", "Related but insufficient evidence."),
            make_results("alternative.txt", "Still insufficient evidence."),
        ]
    )
    evaluator = FakeEvidenceEvaluator(
        [
            EvidenceAssessment(
                sufficient=False,
                reason="The first retrieval is incomplete.",
            ),
            EvidenceAssessment(
                sufficient=False,
                reason="The alternative retrieval is also incomplete.",
            ),
        ]
    )
    planner = FakeQueryPlanner("alternative retrieval query")
    answer_chain = FakeAnswerChain()
    graph = RAGGraph(
        retriever=retriever,
        rag_chain=answer_chain,
        query_planner=planner,
        evidence_evaluator=evaluator,
    )

    response = asyncio.run(
        graph.answer(
            query="What is established by these documents?",
            thread_id="evidence-retry-abstain-test",
        )
    )

    assert response.answer == (
        "اطلاعات مرتبط و کافی در منابع بازیابی‌شده پیدا نشد."
    )
    assert response.sources == []
    assert len(retriever.calls) == 2
    assert len(evaluator.calls) == 2
    assert len(planner.calls) == 1
    assert answer_chain.calls == []


def test_evaluator_failure_does_not_allow_unsupported_generation():
    retriever = FakeRetriever(
        result_sets=[
            make_results("initial.txt", "Possibly incomplete evidence."),
            make_results("alternative.txt", "Still not verified as sufficient."),
        ]
    )
    evaluator = FakeEvidenceEvaluator(
        [
            RuntimeError("Simulated evaluator failure."),
            EvidenceAssessment(
                sufficient=False,
                reason="The evidence remains insufficient.",
            ),
        ]
    )
    planner = FakeQueryPlanner("alternative retrieval query")
    answer_chain = FakeAnswerChain()
    graph = RAGGraph(
        retriever=retriever,
        rag_chain=answer_chain,
        query_planner=planner,
        evidence_evaluator=evaluator,
    )

    response = asyncio.run(
        graph.answer(
            query="What can be concluded from the sources?",
            thread_id="evidence-evaluator-error-test",
        )
    )

    assert response.answer == (
        "اطلاعات مرتبط و کافی در منابع بازیابی‌شده پیدا نشد."
    )
    assert len(retriever.calls) == 2
    assert len(evaluator.calls) == 2
    assert len(planner.calls) == 1
    assert answer_chain.calls == []