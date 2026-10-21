from app.models.schemas import (
    ContentRequest,
    NormalizedContent,
)

from app.services.language_service import (
    detect_language,
)

from app.services.url_service import (
    fetch_url_content,
)


async def normalize_content(
    request: ContentRequest,
) -> NormalizedContent:
    """Normalize text or URL input into a common internal representation."""

    if request.text is not None:
        text = request.text

        return NormalizedContent(
            content=text,
            source_type="text",
            source=request.source,
            task=request.task,
            instruction=request.instruction,
            source_language=detect_language(text),
        )

    url = str(request.url)

    content = await fetch_url_content(url)

    return NormalizedContent(
        content=content,
        source_type="url",
        source=url,
        task=request.task,
        instruction=request.instruction,
        source_language=detect_language(content),
    )