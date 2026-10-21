import asyncio

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.rag.rag_chain import RAGAnswer, RAGChain
from app.rag.rag_graph import RAGGraph


class FakeRAGModel:
    """Return deterministic answers without calling Ollama."""

    def __init__(self):
        self.calls = 0

    def with_structured_output(self, _schema):
        async def invoke(_input):
            self.calls += 1
            return RAGAnswer(
                answer="پاسخ آزمایشی بر اساس منابع بازیابی‌شده"
            )

        return RunnableLambda(invoke)


class SequenceRetriever:
    """Return predefined retrieval results for consecutive searches."""

    def __init__(self, result_batches):
        self.result_batches = list(result_batches)
        self.calls = 0
        self.queries = []
        self.k_values = []
        self.score_thresholds = []

    async def retrieve(
        self,
        query,
        k=4,
        score_threshold=None,
    ):
        self.queries.append(query)
        self.k_values.append(k)
        self.score_thresholds.append(score_threshold)

        call_index = self.calls
        self.calls += 1

        if call_index >= len(self.result_batches):
            return []

        return self.result_batches[call_index]


class FakeResearchQueryPlanner:
    """Produce a predictable alternative query for agent tests."""

    def __init__(self, alternative_query):
        self.alternative_query = alternative_query
        self.calls = []

    async def create_alternative_query(
        self,
        question,
        attempted_query,
    ):
        self.calls.append(
            {
                "question": question,
                "attempted_query": attempted_query,
            }
        )
        return self.alternative_query


def build_test_graph(retriever, planner, model):
    """Construct the RAG graph with local test doubles."""

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    return RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
        query_planner=planner,
    )


def test_research_agent_retries_with_alternative_query():
    """Verify weak initial evidence triggers a successful second search."""

    question = "چرا درختان به خنک شدن شهر کمک می‌کنند؟"
    alternative_query = (
        "اثر سایه درختان شهری و تبخیر آب برگ‌ها بر دمای محیط"
    )

    document = Document(
        page_content=(
            "درختان با ایجاد سایه و تبخیر آب از برگ‌ها "
            "به کاهش دمای محیط شهری کمک می‌کنند."
        ),
        metadata={
            "source": "urban-trees.txt",
            "source_type": "text",
            "chunk_index": 1,
            "chunk_count": 1,
        },
    )

    # The first result is below the threshold; the second is relevant.
    retriever = SequenceRetriever(
        [
            [(document, 0.20)],
            [(document, 0.72)],
        ]
    )
    planner = FakeResearchQueryPlanner(alternative_query)
    model = FakeRAGModel()
    graph = build_test_graph(
        retriever=retriever,
        planner=planner,
        model=model,
    )

    response = asyncio.run(
        graph.answer(
            query=question,
            k=4,
            score_threshold=0.4,
            thread_id="research-retry-success",
        )
    )

    assert retriever.queries == [
        question,
        alternative_query,
    ]
    assert retriever.k_values == [4, 8]
    assert retriever.score_thresholds == [0.4, 0.4]
    assert len(planner.calls) == 1
    assert model.calls == 1

    assert response.sources
    assert response.sources[0].source == "urban-trees.txt"


def test_research_agent_abstains_after_both_searches_fail():
    """Verify the agent stops after one unsuccessful alternative search."""

    question = "این رویداد چه ارتباطی با اسناد قبلی دارد؟"
    alternative_query = "رابطه رویداد با محتوای اسناد قبلی"

    retriever = SequenceRetriever(
        [
            [],
            [],
        ]
    )
    planner = FakeResearchQueryPlanner(alternative_query)
    model = FakeRAGModel()
    graph = build_test_graph(
        retriever=retriever,
        planner=planner,
        model=model,
    )

    response = asyncio.run(
        graph.answer(
            query=question,
            k=4,
            score_threshold=0.4,
            thread_id="research-retry-failure",
        )
    )

    assert retriever.calls == 2
    assert retriever.queries == [
        question,
        alternative_query,
    ]
    assert len(planner.calls) == 1
    assert model.calls == 0

    assert response.sources == []
    assert response.answer == (
        "اطلاعات مرتبط و کافی در منابع "
        "بازیابی‌شده پیدا نشد."
    )


def test_research_agent_stops_when_alternative_is_unchanged():
    """Verify an unchanged query is not searched a second time."""

    question = "چرا درختان به خنک شدن شهر کمک می‌کنند؟"

    retriever = SequenceRetriever(
        [
            [],
        ]
    )
    planner = FakeResearchQueryPlanner(question)
    model = FakeRAGModel()
    graph = build_test_graph(
        retriever=retriever,
        planner=planner,
        model=model,
    )

    response = asyncio.run(
        graph.answer(
            query=question,
            score_threshold=0.4,
            thread_id="research-identical-query",
        )
    )

    assert retriever.calls == 1
    assert len(planner.calls) == 1
    assert model.calls == 0
    assert response.sources == []