import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)

from app.models.schemas import (
    ContentRequest,
    DocumentIngestionResponse,
    NormalizedContent,
    TaskName,
)

from app.services.document_ingestion_service import (
    DocumentIngestionService,
)

from app.services.input_service import (
    normalize_content,
)

from app.services.language_service import (
    detect_language,
)

from app.services.pdf_service import (
    extract_text_from_pdf,
)

from app.services.guardrail_service import (
    InputGuardrail,
    SecurityViolationError,
)

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["documents"],
)


# Initial upload limit to protect local disk resources.
# This can later be moved into application settings.
MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024

UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024


def get_document_ingestion_service(
    request: Request,
) -> DocumentIngestionService:
    """Return the application-wide document ingestion service."""

    return request.app.state.document_ingestion_service


@router.post(
    "/ingest",
    response_model=DocumentIngestionResponse,
)
async def ingest_content_endpoint(
    content_request: ContentRequest,
    ingestion_service: Annotated[
        DocumentIngestionService,
        Depends(get_document_ingestion_service),
    ],
) -> DocumentIngestionResponse:
    """Ingest text or URL content into the vector database."""

    # Reuse the existing normalization pipeline for text and URLs.
    normalized_content = await normalize_content(
        content_request
    )

    # A URL may produce empty content even when its URL is valid.
    if not normalized_content.content.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable content was found.",
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

    return await ingestion_service.ingest(
        normalized_content
    )


@router.post(
    "/ingest-pdf",
    response_model=DocumentIngestionResponse,
)
async def ingest_pdf_endpoint(
    ingestion_service: Annotated[
        DocumentIngestionService,
        Depends(get_document_ingestion_service),
    ],
    file: UploadFile = File(...),
    task: TaskName | None = Form(default=None),
    instruction: str | None = Form(default=None),
) -> DocumentIngestionResponse:
    """Extract text from a PDF and ingest it into the vector database."""

    temp_path: str | None = None

    try:
        # Validate the filename before processing the uploaded file.
        suffix = Path(file.filename or "").suffix.lower()

        if suffix != ".pdf":
            raise HTTPException(
                status_code=415,
                detail="Only PDF files are supported.",
            )

        # Stream the upload to disk instead of loading the entire file
        # into memory at once.
        total_bytes = 0

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

            while True:
                chunk = await file.read(
                    UPLOAD_CHUNK_SIZE_BYTES
                )

                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_PDF_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "The PDF exceeds the maximum "
                            "allowed size of 20 MiB."
                        ),
                    )

                temp_file.write(chunk)

        # Reuse the existing PDF extraction implementation.
        text = extract_text_from_pdf(
            temp_path
        )

        if not text or not text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No extractable text was found in the PDF."
                ),
            )

        # Build the same normalized data structure used by other inputs.
        normalized_content = NormalizedContent(
            content=text,
            source_type="pdf",
            source=Path(file.filename).name,
            task=task,
            instruction=(
                instruction.strip()
                if instruction and instruction.strip()
                else None
            ),
            source_language=detect_language(text),
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

        return await ingestion_service.ingest(
            normalized_content
        )

    finally:
        # Close the upload and remove the temporary file even if
        # validation, extraction, or ingestion fails.
        await file.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)