from app.models.schemas import (
    ContentResponse,
    NormalizedContent,
)


async def analyze_content(
    content: NormalizedContent,
) -> ContentResponse:

    return ContentResponse(
        input_type=content.source_type,
        title="Content analysis result",
        summary=f"Mock analysis: {content.content}",
        importance=50,
        category="other",
    )