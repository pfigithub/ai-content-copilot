import pymupdf
import pytest

from app.services.pdf_service import (
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