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