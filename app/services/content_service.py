from app.models.schemas import ContentResponse


async def analyze_content(text: str) -> ContentResponse:
    return ContentResponse(
        input_type="text",
        title="Content analysis result",
        summary=f"Mock analysis: {text}",
        importance=50,
        category="other",
    )