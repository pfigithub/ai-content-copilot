import asyncio
import json

import httpx

from app.providers.ollama_provider import OllamaProvider


class FakeResponse:

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeAsyncClient:

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        tb,
    ):
        pass

    async def post(
        self,
        *args,
        **kwargs,
    ):
        return FakeResponse(
            {
                "message": {
                    "content": json.dumps(
                        {
                            "title": "Test title",
                            "summary": "Test summary",
                            "importance": 70,
                            "category": "technology",
                        }
                    )
                }
            }
        )


def test_ollama_provider_returns_valid_result(
    monkeypatch,
):

    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda: FakeAsyncClient(),
    )

    provider = OllamaProvider()

    result = asyncio.run(
        provider.analyze_content(
            "This is test content.",
        )
    )

    assert result == {
        "title": "Test title",
        "summary": "Test summary",
        "importance": 70,
        "category": "technology",
    }