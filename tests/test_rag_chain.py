import asyncio

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from app.rag.rag_chain import (
    RAGAnswer,
    RAGChain,
)


class FakeRetriever:

    def __init__(
        self,
        results,
    ):

        self.results = results
        self.calls = []

    async def retrieve(
        self,
        query,
        k=4,
        score_threshold=None,
    ):

        self.calls.append(
            {
                "query": query,
                "k": k,
                "score_threshold": (
                    score_threshold
                ),
            }
        )

        return self.results


class FakeStructuredRunnable(
    Runnable,
):

    def __init__(
        self,
        result,
        captured_inputs,
    ):

        self.result = result
        self.captured_inputs = (
            captured_inputs
        )

    def invoke(
        self,
        input,
        config=None,
        **kwargs,
    ):

        self.captured_inputs.append(
            input
        )

        return self.result

    async def ainvoke(
        self,
        input,
        config=None,
        **kwargs,
    ):

        self.captured_inputs.append(
            input
        )

        return self.result


class FakeModel:

    def __init__(self):

        self.captured_inputs = []

    def with_structured_output(
        self,
        schema,
    ):

        assert schema is RAGAnswer

        return (
            FakeStructuredRunnable(
                result=RAGAnswer(
                    answer=(
                        "آبیاری قطره‌ای می‌تواند "
                        "مصرف آب کشاورزی را کاهش دهد."
                    )
                ),
                captured_inputs=(
                    self.captured_inputs
                ),
            )
        )


def test_rag_chain_builds_grounded_answer():

    retriever = FakeRetriever(
        [
            (
                Document(
                    page_content=(
                        "آبیاری قطره‌ای می‌تواند "
                        "مصرف آب را کاهش دهد."
                    ),
                    metadata={
                        "source": "agriculture.txt",
                        "source_type": "text",
                    },
                ),
                0.91,
            ),
        ]
    )

    model = FakeModel()

    chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    response = asyncio.run(
        chain.answer(
            query=(
                "چطور مصرف آب کشاورزی "
                "را کاهش دهیم؟"
            ),
            k=3,
            score_threshold=0.5,
        )
    )

    assert (
        response.answer
        == "آبیاری قطره‌ای می‌تواند "
        "مصرف آب کشاورزی را کاهش دهد."
    )

    assert len(response.sources) == 1

    assert (
        response.sources[0].source
        == "agriculture.txt"
    )

    assert (
        response.sources[0].score
        == 0.91
    )

    assert (
        retriever.calls[0]
        == {
            "query": (
                "چطور مصرف آب کشاورزی "
                "را کاهش دهیم؟"
            ),
            "k": 3,
            "score_threshold": 0.5,
        }
    )

    captured = (
        model.captured_inputs[0]
    )

    messages = captured.to_messages()

    assert (
        messages[0].type
        == "system"
    )

    assert (
        messages[1].type
        == "human"
    )

    assert (
        messages[1].content
        .find(
            "چطور مصرف آب کشاورزی "
            "را کاهش دهیم؟"
        )
        != -1
    )

    assert (
        messages[1].content
        .find(
            "آبیاری قطره‌ای"
        )
        != -1
    )

    assert (
        messages[1].content.find(
            "agriculture.txt"
        )
        != -1
    )
    
def test_rag_chain_returns_chunk_metadata():
    document = Document(
        page_content="آبیاری قطره‌ای مصرف آب را کاهش می‌دهد.",
        metadata={
            "source": "agriculture.txt",
            "source_type": "evaluation",
            "chunk_index": 2,
            "chunk_count": 5,
        },
    )

    retriever = FakeRetriever(
        results=[
            (document, 0.72)
        ]
    )

    model = FakeModel()

    rag = RAGChain(
        retriever=retriever,
        model=model,
    )

    response = asyncio.run(
        rag.answer(
            query="چطور مصرف آب را کاهش دهیم؟"
        )
    )

    assert len(response.sources) == 1

    source = response.sources[0]

    assert source.source == "agriculture.txt"
    assert source.source_type == "evaluation"
    assert source.chunk_index == 2
    assert source.chunk_count == 5
    assert source.score == 0.72