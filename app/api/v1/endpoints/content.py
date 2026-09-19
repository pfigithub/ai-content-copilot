import os
import tempfile

from fastapi import APIRouter, File, UploadFile

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
):
    normalized_content = await normalize_content(
        request
    )

    return await analyze_content(
        normalized_content
    )


@router.post(
    "/analyze-pdf",
    response_model=ContentResponse,
)
async def analyze_pdf_endpoint(
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
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                temp_file.write(chunk)

        finally:
            await file.close()

    try:
        text = extract_text_from_pdf(
            temp_path
        )

        return await analyze_content(
            type(
                "NormalizedContentData",
                (),
                {
                    "content": text,
                    "source_type": "pdf",
                    "source": file.filename,
                },
            )()
        )

    finally:
        os.remove(temp_path)