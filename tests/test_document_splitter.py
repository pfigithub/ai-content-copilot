import pytest

from langchain_core.documents import (
    Document,
)

from app.retrieval.document_splitter import (
    split_document,
)


def test_split_document_returns_langchain_documents():

    text = (
        "Artificial intelligence is changing "
        "content creation. "
    ) * 100

    chunks = split_document(
        text=text,
        source="test.txt",
        source_type="text",
    )

    assert len(chunks) > 1

    assert all(
        isinstance(chunk, Document)
        for chunk in chunks
    )


def test_split_document_preserves_metadata():

    text = (
        "Artificial intelligence can support "
        "research and content production. "
    ) * 50

    chunks = split_document(
        text=text,
        source="example.pdf",
        source_type="pdf",
    )

    assert len(chunks) > 1

    for chunk in chunks:

        assert (
            chunk.metadata["source"]
            == "example.pdf"
        )

        assert (
            chunk.metadata["source_type"]
            == "pdf"
        )


def test_split_document_rejects_empty_text():

    with pytest.raises(ValueError):

        split_document(
            text="   ",
        )


def test_split_document_validates_chunk_size():

    with pytest.raises(ValueError):

        split_document(
            text="Some content.",
            chunk_size=0,
        )


def test_split_document_validates_chunk_overlap():

    with pytest.raises(ValueError):

        split_document(
            text="Some content.",
            chunk_size=100,
            chunk_overlap=100,
        )
        
def test_split_document_adds_chunk_metadata():
    text = "این یک متن آزمایشی نسبتاً طولانی است. " * 100

    chunks = split_document(
        text=text,
        source="example.txt",
        source_type="test",
        chunk_size=200,
        chunk_overlap=20,
    )

    assert len(chunks) > 1

    for index, chunk in enumerate(chunks, start=1):
        assert chunk.metadata["source"] == "example.txt"
        assert chunk.metadata["source_type"] == "test"
        assert chunk.metadata["chunk_index"] == index
        assert chunk.metadata["chunk_count"] == len(chunks)