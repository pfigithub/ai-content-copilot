import asyncio

import pytest

from langchain_core.documents import (
    Document,
)

from app.retrieval.retriever import (
    ContentRetriever,
)


class FakeQdrantStore:

    def __init__(self):

        self.queries = []

    async def similarity_search_with_score(
        self,
        query,
        k=4,
        score_threshold=None,
    ):

        self.queries.append(
            {
                "query": query,
                "k": k,
                "score_threshold": (
                    score_threshold
                ),
            }
        )

        return [
            (
                Document(
                    page_content="Relevant content.",
                    metadata={
                        "source": "test.txt"
                    },
                ),
                0.92,
            ),
            (
                Document(
                    page_content="Another relevant part.",
                    metadata={
                        "source": "test.txt"
                    },
                ),
                0.81,
            ),
        ]


def test_retriever_returns_documents_and_scores():

    store = FakeQdrantStore()

    retriever = ContentRetriever(
        store=store
    )

    results = asyncio.run(
        retriever.retrieve(
            query="content retrieval",
            k=2,
            score_threshold=0.7,
        )
    )

    assert len(results) == 2

    document, score = results[0]

    assert isinstance(
        document,
        Document,
    )

    assert score == 0.92

    assert (
        store.queries[0]
        == {
            "query": "content retrieval",
            "k": 2,
            "score_threshold": 0.7,
        }
    )


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_retriever_rejects_empty_query(
    query,
):

    retriever = ContentRetriever(
        store=FakeQdrantStore()
    )

    with pytest.raises(
        ValueError
    ):

        asyncio.run(
            retriever.retrieve(
                query=query
            )
        )


def test_retriever_rejects_invalid_k():

    retriever = ContentRetriever(
        store=FakeQdrantStore()
    )

    with pytest.raises(
        ValueError
    ):

        asyncio.run(
            retriever.retrieve(
                query="test",
                k=0,
            )
        )