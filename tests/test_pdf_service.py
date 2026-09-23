import pymupdf
import pytest

from app.services.pdf_service import (
    _order_line_words,
    extract_text_from_pdf,
)


def create_test_pdf(path):

    document = pymupdf.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "This is a test PDF containing readable text "
        "for the AI Content Copilot project.",
    )

    document.save(path)

    document.close()


def test_extract_text_from_pdf(tmp_path):

    pdf_path = (
        tmp_path / "test.pdf"
    )

    create_test_pdf(pdf_path)

    text = extract_text_from_pdf(
        pdf_path
    )

    assert "test PDF" in text


def test_rejects_unreadable_pdf(tmp_path):

    pdf_path = (
        tmp_path / "empty.pdf"
    )

    document = pymupdf.open()

    document.new_page()

    document.save(pdf_path)

    document.close()

    with pytest.raises(ValueError):

        extract_text_from_pdf(
            pdf_path
        )


def test_orders_rtl_words_from_right_to_left():

    words = [
        (
            50.0,
            100.0,
            100.0,
            115.0,
            "مذاکره",
            0,
            0,
            0,
        ),
        (
            120.0,
            100.0,
            160.0,
            115.0,
            "اصول",
            0,
            0,
            1,
        ),
    ]

    result = _order_line_words(
        words
    )

    assert result == [
        "اصول",
        "مذاکره",
    ]


def test_preserves_ltr_word_order():

    words = [
        (
            10.0,
            100.0,
            50.0,
            115.0,
            "Artificial",
            0,
            0,
            0,
        ),
        (
            60.0,
            100.0,
            100.0,
            115.0,
            "intelligence",
            0,
            0,
            1,
        ),
    ]

    result = _order_line_words(
        words
    )

    assert result == [
        "Artificial",
        "intelligence",
    ]