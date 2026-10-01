from bs4 import BeautifulSoup

from app.exceptions import URLFetchError


def extract_text_from_html(
    html: str,
) -> str:

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    for tag in soup(
        ["script", "style", "noscript"]
    ):
        tag.decompose()

    content = soup.find("article")

    if content is None:
        content = soup.find("main")

    if content is None:
        content = soup.body

    if content is None:
        content = soup

    text = content.get_text(
        " ",
        strip=True,
    )

    if not text:
        raise URLFetchError()

    return text