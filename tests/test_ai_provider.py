import asyncio

from app.providers.ai_provider import AIProvider
from app.providers.mock_provider import MockAIProvider


def test_mock_provider_implements_ai_provider():

    provider = MockAIProvider()

    assert isinstance(
        provider,
        AIProvider,
    )

    result = asyncio.run(
        provider.analyze_content(
            "This is a test content."
        )
    )

    assert result == {
        "title": "Content analysis result",
        "summary": (
            "Mock analysis: "
            "This is a test content."
        ),
        "importance": 50,
        "category": "other",
    }