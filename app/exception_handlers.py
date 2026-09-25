from fastapi import (
    Request,
)
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.responses import (
    JSONResponse,
)

from app.exceptions import (
    AIProviderResponseError,
    AIProviderTimeoutError,
    AIProviderUnavailableError,
    AppError,
    PDFExtractionError,
    URLFetchError,
)


ERROR_STATUS_CODES = {
    "AI_PROVIDER_UNAVAILABLE": 503,
    "AI_PROVIDER_TIMEOUT": 504,
    "AI_PROVIDER_RESPONSE_ERROR": 502,
    "URL_FETCH_FAILED": 502,
    "PDF_EXTRACTION_FAILED": 422,
}


async def app_error_handler(
    request: Request,
    exc: AppError,
):

    status_code = (
        ERROR_STATUS_CODES.get(
            exc.code,
            500,
        )
    )

    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
            }
        },
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
):

    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": (
                    "The request data is invalid."
                ),
            }
        },
    )