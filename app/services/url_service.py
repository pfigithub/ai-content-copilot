import httpx

from app.exceptions import (
    URLFetchError,
)
from app.services.html_service import (
    extract_text_from_html,
)


DEFAULT_TIMEOUT = 20.0


async def fetch_url_content(
    url: str,
) -> str:

    try:

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=DEFAULT_TIMEOUT,
        ) as client:

            response = await client.get(
                url
            )

            response.raise_for_status()

    except httpx.HTTPError as exc:

        raise URLFetchError() from exc

    return extract_text_from_html(
        response.text
    )