from pathlib import Path

import pymupdf


def extract_text_from_pdf(
    file_path: str | Path,
) -> str:

    try:
        with pymupdf.open(file_path) as document:
            pages = [
                page.get_text("text")
                for page in document
            ]

    except Exception as exc:
        raise ValueError(
            "The PDF could not be opened or processed."
        ) from exc

    text = "\n".join(pages).strip()

    if len(text) < 20:
        raise ValueError(
            "The PDF does not contain enough readable text."
        )

    if "\ufffd" in text:
        raise ValueError(
            "The PDF text could not be decoded reliably."
        )

    private_use_chars = sum(
        1
        for char in text
        if "\ue000" <= char <= "\uf8ff"
    )

    if (
        private_use_chars > 0
        and private_use_chars / len(text) > 0.05
    ):
        raise ValueError(
            "The PDF text extraction appears unreliable."
        )

    return text