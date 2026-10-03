from langchain_core.documents import (
    Document,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120


def split_document(
    text: str,
    source: str | None = None,
    source_type: str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:

    if not text or not text.strip():
        raise ValueError(
            "Document text cannot be empty."
        )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "chunk_overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    document = Document(
        page_content=text,
        metadata={
            "source": source,
            "source_type": source_type,
        },
    )

    splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )

    return splitter.split_documents(
        [document]
    )