import os
import tempfile
from typing import Annotated

from fastapi import (
    APIRouter, Depends, File, Form, UploadFile, HTTPException
)

from app.dependencies import get_ai_provider
from app.models.schemas import (
    AnalysisContentResponse, ContentRequest, IdeasContentResponse,
    NormalizedContent, SummaryContentResponse, TaskName,
)
from app.providers.ai_provider import AIProvider
from app.services.content_service import analyze_content
from app.services.input_service import normalize_content
from app.services.pdf_service import extract_text_from_pdf
from app.services.language_service import detect_language
from app.services.guardrail_service import InputGuardrail, OutputGuardrail, SecurityViolationError

router = APIRouter(prefix="/api/v1/content", tags=["content"])

ContentResponseModel = (
    SummaryContentResponse | AnalysisContentResponse | IdeasContentResponse
)

MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024

@router.post("/analyze-content", response_model=ContentResponseModel)
async def analyze_content_endpoint(
    request: ContentRequest,
    ai_provider: Annotated[AIProvider, Depends(get_ai_provider)],
):
    normalized_content = await normalize_content(request)
    
    input_guard = InputGuardrail()
    try:
        await input_guard.verify_safe_content(normalized_content.content)
    except SecurityViolationError as exc:
        raise HTTPException(status_code=400, detail=exc.message)

    result = await analyze_content(normalized_content, ai_provider)
    
    # Enforce Output Guardrail on generated content before returning to user
    output_guard = OutputGuardrail()
    try:
        text_to_check = getattr(result, "summary", getattr(result, "analysis", ""))
        await output_guard.verify_safe_output(text_to_check)
    except SecurityViolationError as exc:
        raise HTTPException(status_code=400, detail="Output generation blocked by safety policies.")
        
    return result

@router.post("/analyze-pdf", response_model=ContentResponseModel)
async def analyze_pdf_endpoint(
    ai_provider: Annotated[AIProvider, Depends(get_ai_provider)],
    file: UploadFile = File(...),
    task: TaskName | None = Form(default=None),
    instruction: str | None = Form(default=None),
):
    suffix = os.path.splitext(file.filename or "")[1]
    total_bytes = 0
    temp_path = None
    
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
            temp_path = temp_file.name
            while True:
                chunk = await file.read(UPLOAD_CHUNK_SIZE_BYTES)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_PDF_SIZE_BYTES:
                    # Closing file explicitly before raising so the finally block can safely delete it
                    temp_file.close()
                    raise HTTPException(
                        status_code=413, detail="The PDF exceeds the maximum allowed size of 20 MiB."
                    )
                temp_file.write(chunk)

        text, _ = extract_text_from_pdf(temp_path)
        detected_lang = detect_language(text)

        input_guard = InputGuardrail()
        try:
            await input_guard.verify_safe_content(text)
        except SecurityViolationError as exc:
            raise HTTPException(status_code=400, detail=exc.message)

        normalized_content = NormalizedContent(
            content=text,
            source_type="pdf",
            source=file.filename,
            task=task,
            instruction=(instruction.strip() if instruction else None),
            source_language=detected_lang,
        )

        result = await analyze_content(normalized_content, ai_provider)
        
        output_guard = OutputGuardrail()
        try:
            text_to_check = getattr(result, "summary", getattr(result, "analysis", ""))
            await output_guard.verify_safe_output(text_to_check)
        except SecurityViolationError as exc:
            raise HTTPException(status_code=400, detail="Output generation blocked by safety policies.")
            
        return result
        
    finally:
        await file.close()
        # Safe cleanup even if interrupted by size exception
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)