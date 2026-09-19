from bs4 import BeautifulSoup


def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(
        ["script", "style", "noscript"]
    ):
        tag.decompose()

    text = soup.get_text(
        " ",
        strip=True,
    )

    if not text:
        raise ValueError(
            "No readable text was found in the HTML content."
        )

    return text