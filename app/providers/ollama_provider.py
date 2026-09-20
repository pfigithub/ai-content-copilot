import json

import httpx
from pydantic import ValidationError

from app.config import settings
from app.models.schemas import AIContentResult
from app.providers.ai_provider import AIProvider


class OllamaProvider(AIProvider):

    async def analyze_content(
        self,
        content: str,
        task: str | None = None,
    ) -> dict:

        prompt = f"""
تو یک دستیار هوشمند تحلیل و پردازش محتوا هستی.

محتوای زیر را تحلیل کن:

{content}

قوانین:
- اطلاعاتی که در محتوا وجود ندارند را جعل نکن.
- عنوان کوتاه و روشن تولید کن.
- یک خلاصه دقیق و کاربردی ارائه بده.
- importance را بین 1 تا 100 قرار بده.
- category را متناسب با موضوع انتخاب کن.
- خروجی فقط JSON معتبر باشد.
- هیچ توضیحی قبل یا بعد از JSON ننویس.

ساختار خروجی:

{{
    "title": "عنوان",
    "summary": "خلاصه",
    "importance": 50,
    "category": "other"
}}
"""

        async with httpx.AsyncClient() as client:

            response = await client.post(
                f"{settings.ollama_base_url}/api/chat",
                json={
                    "model": settings.ollama_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    "stream": False,
                    "format": (
                        AIContentResult.model_json_schema()
                    ),
                    "options": {
                        "temperature": 0,
                    },
                },
                timeout=120.0,
            )

        response.raise_for_status()

        data = response.json()

        content = data["message"]["content"]

        try:
            result = json.loads(content)

        except json.JSONDecodeError as exc:
            raise ValueError(
                "Ollama returned invalid JSON."
            ) from exc

        try:
            validated_result = (
                AIContentResult.model_validate(result)
            )

        except ValidationError as exc:
            raise ValueError(
                "Ollama returned an invalid structure."
            ) from exc

        return validated_result.model_dump()