from app.models.schemas import (
    ContentRequest,
    NormalizedContent,
)

from app.services.url_service import fetch_url_content


async def normalize_content(
    request: ContentRequest,
) -> NormalizedContent:

    if request.text is not None:
        return NormalizedContent(
            content=request.text,
            source_type="text",
        )

    url = str(request.url)

    content = await fetch_url_content(url)

    return NormalizedContent(
        content=content,
        source_type="url",
        source=url,
    )