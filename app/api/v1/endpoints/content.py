
import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.dependencies import get_ai_provider, get_current_user
from app.models.schemas import (
    AnalysisContentResponse,
    ContentRequest,
    IdeasContentResponse,
    NormalizedContent,
    SummaryContentResponse,
    TaskName,
)
from app.providers.ai_provider import AIProvider
from app.services.content_service import analyze_content
from app.services.guardrail_service import (
    InputGuardrail,
    OutputGuardrail,
    SecurityViolationError,
)
from app.services.input_service import normalize_content
from app.services.language_service import detect_language
from app.services.pdf_service import extract_text_from_pdf


router = APIRouter(
    prefix="/api/v1/content",
    tags=["content"],
    # Protect every current and future content endpoint by default.
    dependencies=[Depends(get_current_user)],
)

ContentResponseModel = (
    SummaryContentResponse
    | AnalysisContentResponse
    | IdeasContentResponse
)

MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


async def verify_input(text: str) -> None:
    """Inspect source content and user instructions before model processing."""
    guardrail = InputGuardrail()

    try:
        await guardrail.verify_safe_content(text)
    except SecurityViolationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=exc.message,
        ) from exc


async def verify_output(result: ContentResponseModel) -> None:
    """Inspect every structured output field before returning it to the client."""
    guardrail = OutputGuardrail()

    try:
        # JSON serialization includes title, summary, analysis, ideas, and metadata.
        await guardrail.verify_safe_output(result.model_dump_json())
    except SecurityViolationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Output generation was blocked by the security policy.",
        ) from exc


def combine_untrusted_input(content: str, instruction: str | None) -> str:
    """Combine source text and user instructions for security inspection."""
    parts = [content]

    if instruction:
        parts.append(instruction)

    return "\n\n".join(parts)


@router.post("/analyze-content", response_model=ContentResponseModel)
async def analyze_content_endpoint(
    request: ContentRequest,
    ai_provider: Annotated[AIProvider, Depends(get_ai_provider)],
) -> ContentResponseModel:
    """Analyze text or URL content after authentication and input screening."""

    normalized_content = await normalize_content(request)

    await verify_input(
        combine_untrusted_input(
            normalized_content.content,
            normalized_content.instruction,
        )
    )

    result = await analyze_content(normalized_content, ai_provider)

    await verify_output(result)

    return result


@router.post("/analyze-pdf", response_model=ContentResponseModel)
async def analyze_pdf_endpoint(
    ai_provider: Annotated[AIProvider, Depends(get_ai_provider)],
    file: UploadFile = File(...),
    task: TaskName | None = Form(default=None),
    instruction: str | None = Form(default=None),
) -> ContentResponseModel:
    """Analyze a temporary PDF and remove its temporary copy on every exit path."""

    filename = file.filename or ""

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    temp_path: str | None = None
    total_bytes = 0

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE_BYTES)

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_PDF_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="The PDF exceeds the maximum allowed size of 20 MiB.",
                    )

                temp_file.write(chunk)

        if total_bytes == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The uploaded PDF is empty.",
            )

        text, _pages = extract_text_from_pdf(temp_path)

        if not text or not text.strip():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No extractable text was found in the PDF.",
            )

        normalized_instruction = (
            instruction.strip()
            if instruction and instruction.strip()
            else None
        )

        await verify_input(
            combine_untrusted_input(text, normalized_instruction)
        )

        normalized_content = NormalizedContent(
            content=text,
            source_type="pdf",
            source=Path(filename).name,
            task=task,
            instruction=normalized_instruction,
            source_language=detect_language(text),
        )

        result = await analyze_content(
            normalized_content,
            ai_provider,
        )

        await verify_output(result)

        return result

    finally:
        await file.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
