import httpx

from app.services.html_service import extract_text_from_html


async def fetch_url_content(url: str) -> str:
    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=15.0,
        headers={
            "User-Agent": "AI-Content-Copilot/0.1",
        },
    ) as client:

        response = await client.get(url)

        response.raise_for_status()

        return extract_text_from_html(
            response.text
        )