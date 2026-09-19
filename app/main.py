from fastapi import FastAPI

from app.api.v1.endpoints.content import router as content_router


app = FastAPI(
    title="AI Content Copilot",
    version="0.1.0.dev0",
)


app.include_router(content_router)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "AI Content Copilot",
        "version": "0.1.0.dev0",
    }