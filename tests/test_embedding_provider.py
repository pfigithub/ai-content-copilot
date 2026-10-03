import asyncio

import pytest

from app.retrieval.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)


class FakeOllamaEmbeddings:

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        self.args = args
        self.kwargs = kwargs

    async def aembed_documents(
        self,
        texts,
    ):

        return [
            [0.1, 0.2, 0.3]
            for _ in texts
        ]

    async def aembed_query(
        self,
        text,
    ):

        return [
            0.4,
            0.5,
            0.6,
        ]


def test_embedding_provider_creates_ollama_embeddings(
    monkeypatch,
):

    monkeypatch.setattr(
        "app.retrieval.ollama_embedding_provider.OllamaEmbeddings",
        FakeOllamaEmbeddings,
    )

    provider = (
        OllamaEmbeddingProvider()
    )

    assert (
        provider.embeddings.kwargs["model"]
        == "qwen3-embedding:0.6b"
    )


def test_embed_documents_returns_vectors(
    monkeypatch,
):

    monkeypatch.setattr(
        "app.retrieval.ollama_embedding_provider.OllamaEmbeddings",
        FakeOllamaEmbeddings,
    )

    provider = (
        OllamaEmbeddingProvider()
    )

    texts = [
        "هوش مصنوعی در تولید محتوا.",
        "بازیابی اطلاعات به embedding نیاز دارد.",
    ]

    vectors = asyncio.run(
        provider.embed_documents(
            texts
        )
    )

    assert len(vectors) == 2
    assert all(
        isinstance(vector, list)
        for vector in vectors
    )
    assert all(
        len(vector) == 3
        for vector in vectors
    )


def test_embed_query_returns_vector(
    monkeypatch,
):

    monkeypatch.setattr(
        "app.retrieval.ollama_embedding_provider.OllamaEmbeddings",
        FakeOllamaEmbeddings,
    )

    provider = (
        OllamaEmbeddingProvider()
    )

    vector = asyncio.run(
        provider.embed_query(
            "بازیابی اطلاعات"
        )
    )

    assert isinstance(
        vector,
        list,
    )

    assert len(vector) == 3


@pytest.mark.parametrize(
    "texts",
    [
        [],
        [""],
        ["   "],
    ],
)
def test_embed_documents_rejects_invalid_input(
    monkeypatch,
    texts,
):

    monkeypatch.setattr(
        "app.retrieval.ollama_embedding_provider.OllamaEmbeddings",
        FakeOllamaEmbeddings,
    )

    provider = (
        OllamaEmbeddingProvider()
    )

    if texts == []:

        vectors = asyncio.run(
            provider.embed_documents(
                texts
            )
        )

        assert vectors == []

    else:

        with pytest.raises(
            ValueError
        ):

            asyncio.run(
                provider.embed_documents(
                    texts
                )
            )


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
    ],
)
def test_embed_query_rejects_invalid_input(
    monkeypatch,
    text,
):

    monkeypatch.setattr(
        "app.retrieval.ollama_embedding_provider.OllamaEmbeddings",
        FakeOllamaEmbeddings,
    )

    provider = (
        OllamaEmbeddingProvider()
    )

    with pytest.raises(
        ValueError
    ):

        asyncio.run(
            provider.embed_query(
                text
            )
        )