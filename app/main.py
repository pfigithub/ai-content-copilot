from fastapi import FastAPI


app = FastAPI(
    title="AI Content Copilot",
    version="0.1.0.dev0",
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "AI Content Copilot",
        "version": "0.1.0.dev0",
    }