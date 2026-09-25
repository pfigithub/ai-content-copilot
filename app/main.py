from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from app.exception_handlers import (
    app_error_handler,
    validation_error_handler,
)
from app.exceptions import (
    AppError,
)

from app.api.v1.endpoints.content import router as content_router


app = FastAPI(
    title="AI Content Copilot",
    version="0.1.0.dev0",
)

app.add_exception_handler(
    AppError,
    app_error_handler,
)

app.add_exception_handler(
    RequestValidationError,
    validation_error_handler,
)

app.include_router(content_router)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "AI Content Copilot",
        "version": "0.1.0.dev0",
    }