from fastapi import APIRouter

from app.models.schemas import (
    ContentRequest,
    ContentResponse,
)

from app.services.content_service import (
    analyze_content,
)

from app.services.input_service import (
    normalize_content,
)


router = APIRouter(
    prefix="/api/v1/content",
    tags=["content"],
)


@router.post(
    "/analyze-content",
    response_model=ContentResponse,
)
async def analyze_content_endpoint(
    request: ContentRequest,
):

    normalized_content = await normalize_content(
        request
    )

    return await analyze_content(
        normalized_content
    )