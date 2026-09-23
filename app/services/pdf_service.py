import unicodedata
from pathlib import Path

import pymupdf


def _is_rtl_character(
    character: str,
) -> bool:

    return unicodedata.bidirectional(
        character
    ) in {"R", "AL"}


def _is_rtl_line(
    words: list[tuple],
) -> bool:

    rtl_count = 0
    ltr_count = 0

    for word in words:

        text = word[4]

        for character in text:

            if _is_rtl_character(character):

                rtl_count += 1

            elif (
                character.isalpha()
                and character.isascii()
            ):

                ltr_count += 1

    return (
        rtl_count > 0
        and rtl_count >= ltr_count
    )


def _order_line_words(
    words: list[tuple],
) -> list[str]:

    if _is_rtl_line(words):

        ordered_words = sorted(
            words,
            key=lambda word: word[0],
            reverse=True,
        )

    else:

        ordered_words = sorted(
            words,
            key=lambda word: word[0],
        )

    return [
        word[4]
        for word in ordered_words
    ]


def _extract_page_text(
    page: pymupdf.Page,
) -> str:

    words = page.get_text(
        "words",
        sort=True,
    )

    lines: dict[
        tuple[int, int],
        list[tuple],
    ] = {}

    for word in words:

        block_number = word[5]
        line_number = word[6]

        key = (
            block_number,
            line_number,
        )

        lines.setdefault(
            key,
            [],
        ).append(word)

    output_lines: list[str] = []

    current_block = None

    for (
        block_number,
        line_number,
    ), line_words in lines.items():

        if (
            current_block is not None
            and block_number != current_block
        ):
            output_lines.append("")

        ordered_words = _order_line_words(
            line_words
        )

        line_text = " ".join(
            ordered_words
        ).strip()

        if line_text:
            output_lines.append(
                line_text
            )

        current_block = block_number

    return "\n".join(
        output_lines
    ).strip()


def extract_text_from_pdf(
    file_path: str | Path,
) -> str:

    try:

        with pymupdf.open(
            file_path
        ) as document:

            pages = [
                _extract_page_text(page)
                for page in document
            ]

    except Exception as exc:

        raise ValueError(
            "The PDF could not be opened or processed."
        ) from exc

    text = "\n".join(
        pages
    ).strip()

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
        for character in text
        if "\ue000"
        <= character
        <= "\uf8ff"
    )

    if (
        private_use_chars > 0
        and private_use_chars / len(text) > 0.05
    ):

        raise ValueError(
            "The PDF text extraction appears unreliable."
        )

    return text