import asyncio

import pytest

from app.providers.ai_provider import (
    AIProvider,
)

from app.providers.mock_provider import (
    MockAIProvider,
)


@pytest.mark.parametrize(
    "task, expected_key",
    [
        ("summarize", "summary"),
        ("analyze", "analysis"),
        ("ideas", "ideas"),
    ],
)
def test_mock_provider_supports_tasks(
    task,
    expected_key,
):

    provider = MockAIProvider()

    result = asyncio.run(
        provider.analyze_content(
            "This is test content.",
            task,
        )
    )

    assert isinstance(
        provider,
        AIProvider,
    )

    assert result["title"] != ""
    assert result["category"] == "other"
    assert expected_key in result


def test_mock_provider_uses_default_task():

    provider = MockAIProvider()

    result = asyncio.run(
        provider.analyze_content(
            "This is default test content."
        )
    )

    assert "summary" in result
    assert "importance" in result
    assert "analysis" not in result
    assert "ideas" not in result