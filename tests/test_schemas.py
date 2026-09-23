import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ContentRequest,
    NormalizedContent,
)


def test_content_request_accepts_supported_tasks():

    for task in (
        "summarize",
        "analyze",
        "ideas",
    ):

        request = ContentRequest(
            text="Test content.",
            task=task,
        )

        assert request.task == task


def test_content_request_allows_default_task():

    request = ContentRequest(
        text="Test content."
    )

    assert request.task is None


def test_content_request_rejects_invalid_task():

    with pytest.raises(ValidationError):

        ContentRequest(
            text="Test content.",
            task="translate",
        )


def test_normalized_content_preserves_task():

    content = NormalizedContent(
        content="Test content.",
        source_type="text",
        task="analyze",
    )

    assert content.task == "analyze"
    
def test_normalized_content_stores_source_language():

    content = NormalizedContent(
        content="این یک متن فارسی است.",
        source_type="text",
        task="summarize",
        source_language="fa",
    )

    assert content.source_language == "fa"