import asyncio

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from app.rag.rag_chain import RAGAnswer, RAGChain
from app.rag.rag_graph import RAGGraph


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    async def retrieve(
        self,
        query,
        k=4,
        score_threshold=None,
    ):
        self.calls += 1
        return self.results


class FakeModel:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, _schema):
        async def invoke(_input):
            self.calls += 1

            return RAGAnswer(
                answer=(
                    "پاسخ آزمایشی بر اساس "
                    "منابع بازیابی‌شده"
                )
            )

        return RunnableLambda(invoke)


def test_rag_graph_retrieves_once_and_generates_from_same_results():
    document = Document(
        page_content=(
            "آبیاری قطره‌ای می‌تواند "
            "مصرف آب را کاهش دهد."
        ),
        metadata={
            "source": "agriculture.txt",
            "source_type": "evaluation",
            "chunk_index": 2,
            "chunk_count": 5,
        },
    )

    retriever = FakeRetriever(
        results=[
            (document, 0.72),
        ]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    response = asyncio.run(
        graph.answer(
            query="چطور مصرف آب کشاورزی را کاهش دهیم؟"
        )
    )

    assert retriever.calls == 1
    assert model.calls == 1

    assert response.answer == (
        "پاسخ آزمایشی بر اساس "
        "منابع بازیابی‌شده"
    )

    assert len(response.sources) == 1

    source = response.sources[0]

    assert source.source == "agriculture.txt"
    assert source.source_type == "evaluation"
    assert source.chunk_index == 2
    assert source.chunk_count == 5
    assert source.score == 0.72


def test_rag_graph_abstains_without_results():
    retriever = FakeRetriever(
        results=[]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    response = asyncio.run(
        graph.answer(
            query="سؤال خارج از منابع"
        )
    )

    assert retriever.calls == 1
    assert model.calls == 0

    assert response.sources == []

    assert response.answer == (
        "اطلاعات مرتبط و کافی در منابع "
        "بازیابی‌شده پیدا نشد."
    )
    
def test_rag_graph_streams_retrieve_and_generate_updates():
    document = Document(
        page_content=(
            "آبیاری قطره‌ای می‌تواند "
            "مصرف آب را کاهش دهد."
        ),
        metadata={
            "source": "agriculture.txt",
            "source_type": "evaluation",
            "chunk_index": 2,
            "chunk_count": 5,
        },
    )

    retriever = FakeRetriever(
        results=[
            (document, 0.72),
        ]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    updates = []

    async def collect_updates():
        async for update in graph.stream_steps(
            query="چطور مصرف آب کشاورزی را کاهش دهیم؟"
        ):
            updates.append(update)

    asyncio.run(collect_updates())

    assert len(updates) == 2

    assert list(updates[0].keys()) == [
        "retrieve"
    ]

    assert list(updates[1].keys()) == [
        "generate"
    ]

    assert "results" in updates[0]["retrieve"]
    assert "response" in updates[1]["generate"]

    assert retriever.calls == 1
    assert model.calls == 1
    
def test_rag_graph_persists_state_by_thread():
    retriever = FakeRetriever(
        results=[]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    thread_id = "thread-persistence-test"

    response = asyncio.run(
        graph.answer(
            query="سؤال آزمایشی برای persistence",
            thread_id=thread_id,
        )
    )

    assert response.sources == []

    snapshot = graph.get_state(thread_id)

    assert snapshot.values["query"] == (
        "سؤال آزمایشی برای persistence"
    )

    assert snapshot.values["results"] == []

    assert snapshot.values["response"].answer == (
        "اطلاعات مرتبط و کافی در منابع "
        "بازیابی‌شده پیدا نشد."
    )


def test_rag_graph_keeps_threads_separate():
    retriever = FakeRetriever(
        results=[]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    asyncio.run(
        graph.answer(
            query="سؤال مربوط به thread اول",
            thread_id="thread-1",
        )
    )

    asyncio.run(
        graph.answer(
            query="سؤال مربوط به thread دوم",
            thread_id="thread-2",
        )
    )

    state_1 = graph.get_state(
        "thread-1"
    )

    state_2 = graph.get_state(
        "thread-2"
    )

    assert state_1.values["query"] == (
        "سؤال مربوط به thread اول"
    )

    assert state_2.values["query"] == (
        "سؤال مربوط به thread دوم"
    )