import pytest

from app.exceptions import URLFetchError
from app.services.html_service import (
    extract_text_from_html,
)


def test_extracts_article_content_first():

    html = """
    <html>
        <body>
            <nav>Navigation</nav>

            <article>
                Main article content.
            </article>

            <footer>Footer</footer>
        </body>
    </html>
    """

    text = extract_text_from_html(
        html
    )

    assert "Main article content." in text
    assert "Navigation" not in text
    assert "Footer" not in text


def test_falls_back_to_main_content():

    html = """
    <html>
        <body>
            <main>
                Main section content.
            </main>
        </body>
    </html>
    """

    text = extract_text_from_html(
        html
    )

    assert text == "Main section content."


def test_empty_html_raises_url_fetch_error():

    with pytest.raises(
        URLFetchError
    ):
        extract_text_from_html(
            "<html><body></body></html>"
        )