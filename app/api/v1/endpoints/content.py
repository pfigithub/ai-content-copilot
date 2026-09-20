import os
import tempfile
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    UploadFile,
)

from app.dependencies import get_ai_provider

from app.models.schemas import (
    ContentRequest,
    ContentResponse,
)

from app.providers.ai_provider import AIProvider

from app.services.content_service import (
    analyze_content,
)

from app.services.input_service import (
    normalize_content,
)

from app.services.pdf_service import (
    extract_text_from_pdf,
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
    ai_provider: Annotated[
        AIProvider,
        Depends(get_ai_provider),
    ],
):

    normalized_content = await normalize_content(
        request
    )

    return await analyze_content(
        normalized_content,
        ai_provider,
    )


@router.post(
    "/analyze-pdf",
    response_model=ContentResponse,
)
async def analyze_pdf_endpoint(
    ai_provider: Annotated[
        AIProvider,
        Depends(get_ai_provider),
    ],
    file: UploadFile = File(...),
):

    suffix = os.path.splitext(
        file.filename or ""
    )[1]

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:

        temp_path = temp_file.name

        try:
            while True:
                chunk = await file.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                temp_file.write(chunk)

        finally:
            await file.close()

    try:
        text = extract_text_from_pdf(
            temp_path
        )

        normalized_content = type(
            "NormalizedContentData",
            (),
            {
                "content": text,
                "source_type": "pdf",
                "source": file.filename,
            },
        )()

        return await analyze_content(
            normalized_content,
            ai_provider,
        )

    finally:
        os.remove(temp_path)