import asyncio

from app.models.schemas import ContentRequest
from app.services.input_service import (
    normalize_content,
)


def test_normalize_url_content_detects_language(
    monkeypatch,
):

    async def fake_fetch_url_content(
        url,
    ):

        return (
            "Reliable public transportation can "
            "reduce traffic congestion and improve "
            "urban mobility."
        )

    monkeypatch.setattr(
        "app.services.input_service.fetch_url_content",
        fake_fetch_url_content,
    )

    request = ContentRequest(
        url="https://example.com/article",
        task="summarize",
    )

    result = asyncio.run(
        normalize_content(
            request
        )
    )

    assert (
        result.source_type
        == "url"
    )

    assert (
        result.source
        == "https://example.com/article"
    )

    assert result.task == "summarize"

    assert (
        result.source_language
        == "en"
    )

    assert result.instruction is None


def test_normalize_url_content_preserves_instruction(
    monkeypatch,
):

    async def fake_fetch_url_content(
        url,
    ):

        return (
            "افزایش استفاده از انرژی‌های تجدیدپذیر "
            "می‌تواند وابستگی به سوخت‌های فسیلی را کاهش دهد."
        )

    monkeypatch.setattr(
        "app.services.input_service.fetch_url_content",
        fake_fetch_url_content,
    )

    request = ContentRequest(
        url="https://example.com/persian-article",
        task="summarize",
        instruction="Answer in English.",
    )

    result = asyncio.run(
        normalize_content(
            request
        )
    )

    assert (
        result.source_type
        == "url"
    )

    assert (
        result.source
        == "https://example.com/persian-article"
    )

    assert (
        result.source_language
        == "fa"
    )

    assert (
        result.instruction
        == "Answer in English."
    )