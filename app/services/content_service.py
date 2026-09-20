from app.models.schemas import (
    ContentResponse,
    NormalizedContent,
)

from app.providers.ai_provider import AIProvider


async def analyze_content(
    content: NormalizedContent,
    ai_provider: AIProvider,
) -> ContentResponse:

    ai_result = await ai_provider.analyze_content(
        content.content
    )

    return ContentResponse(
        input_type=content.source_type,
        **ai_result,
    )