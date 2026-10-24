import os
import tempfile
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
    HTTPException
)

from app.dependencies import get_ai_provider

from app.models.schemas import (
    AnalysisContentResponse,
    ContentRequest,
    IdeasContentResponse,
    NormalizedContent,
    SummaryContentResponse,
    TaskName,
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

from app.services.language_service import (
    detect_language,
)

from app.services.guardrail_service import InputGuardrail, SecurityViolationError

router = APIRouter(
    prefix="/api/v1/content",
    tags=["content"],
)


ContentResponseModel = (
    SummaryContentResponse
    | AnalysisContentResponse
    | IdeasContentResponse
)


@router.post(
    "/analyze-content",
    response_model=ContentResponseModel,
)
async def analyze_content_endpoint(
    request: ContentRequest,
    ai_provider: Annotated[
        AIProvider,
        Depends(get_ai_provider),
    ],
):

    normalized_content = (
        await normalize_content(
            request
        )
    )
    
    # Apply security guardrail to the normalized content
    guardrail = InputGuardrail()
    try:
        await guardrail.verify_safe_content(normalized_content.content)
    except SecurityViolationError as exc:
        raise HTTPException(
            status_code=400,
            detail=exc.message,
        )

    return await analyze_content(
        normalized_content,
        ai_provider,
    )


@router.post(
    "/analyze-pdf",
    response_model=ContentResponseModel,
)
async def analyze_pdf_endpoint(
    ai_provider: Annotated[
        AIProvider,
        Depends(get_ai_provider),
    ],
    file: UploadFile = File(...),
    task: TaskName | None = Form(
        default=None
    ),
    instruction: str | None = Form(
        default=None
    ),
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
        
        # بهینه‌سازی: تشخیص زبان یک بار انجام و ذخیره می‌شود
        detected_lang = detect_language(text)

        normalized_content = (
            NormalizedContent(
                content=text,
                source_type="pdf",
                source=file.filename,
                task=task,
                instruction=(
                    instruction.strip()
                    if instruction
                    else None
                ),
                source_language=detected_lang,
            )
        )

        return await analyze_content(
            normalized_content,
            ai_provider,
        )

    finally:

        os.remove(temp_path)