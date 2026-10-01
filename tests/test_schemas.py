import pytest
from pydantic import ValidationError

from app.models.schemas import (
    ContentRequest,
    NormalizedContent,
    IdeasResult,
    SummaryResult,
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

def test_content_request_rejects_whitespace_instruction():

    with pytest.raises(ValidationError):

        ContentRequest(
            text="Valid content.",
            instruction="   ",
        )


def test_content_request_rejects_both_text_and_url():

    with pytest.raises(ValidationError):

        ContentRequest(
            text="Valid content.",
            url="https://example.com/",
        )


def test_content_request_rejects_missing_source():

    with pytest.raises(ValidationError):

        ContentRequest()


@pytest.mark.parametrize(
    "importance",
    [0, 101, -10, 150],
)
def test_summary_result_rejects_invalid_importance(
    importance,
):

    with pytest.raises(ValidationError):

        SummaryResult(
            title="Test",
            summary="Test summary.",
            importance=importance,
            category="technology",
        )


def test_result_rejects_invalid_category():

    with pytest.raises(ValidationError):

        SummaryResult(
            title="Test",
            summary="Test summary.",
            importance=70,
            category="invalid",
        )


@pytest.mark.parametrize(
    "count",
    [2, 8],
)
def test_ideas_result_rejects_invalid_count(
    count,
):

    ideas = [
        f"Idea {index}"
        for index in range(count)
    ]

    with pytest.raises(ValidationError):

        IdeasResult(
            title="Test ideas",
            summary="Test direction.",
            ideas=ideas,
            category="technology",
        )