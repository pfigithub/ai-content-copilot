import re
import unicodedata

from lingua import (
    LanguageDetector,
    LanguageDetectorBuilder,
)


GENERAL_DETECTOR: LanguageDetector = (
    LanguageDetectorBuilder
    .from_all_spoken_languages()
    .build()
)

ARABIC_SCRIPT_DETECTOR: LanguageDetector = (
    LanguageDetectorBuilder
    .from_all_languages_with_arabic_script()
    .build()
)

CYRILLIC_SCRIPT_DETECTOR: LanguageDetector = (
    LanguageDetectorBuilder
    .from_all_languages_with_cyrillic_script()
    .build()
)


MIN_SCRIPT_LETTERS = 40
MIN_SCRIPT_WORDS = 8
MIN_SCRIPT_WORD_RATIO = 0.30


def _get_script(
    char: str,
) -> str | None:

    name = unicodedata.name(
        char,
        "",
    )

    if "ARABIC" in name:
        return "arabic"

    if "CYRILLIC" in name:
        return "cyrillic"

    if "LATIN" in name:
        return "latin"

    return None


def _count_script_letters(
    text: str,
) -> dict[str, int]:

    counts = {
        "arabic": 0,
        "cyrillic": 0,
        "latin": 0,
    }

    for char in text:

        if not char.isalpha():
            continue

        script = _get_script(char)

        if script is not None:
            counts[script] += 1

    return counts


def _extract_script_text(
    text: str,
    script_name: str,
) -> str:

    characters = []

    for char in text:

        if _get_script(char) == script_name:
            characters.append(char)

        else:
            characters.append(" ")

    return "".join(
        characters
    )


def _count_script_words(
    text: str,
    script_name: str,
) -> int:

    script_text = _extract_script_text(
        text,
        script_name,
    )

    return len(
        re.findall(
            r"\S+",
            script_text,
        )
    )


def _detect_from_script(
    text: str,
    script_name: str,
) -> str | None:

    script_text = _extract_script_text(
        text,
        script_name,
    ).strip()

    if not script_text:
        return None

    if script_name == "arabic":

        language = (
            ARABIC_SCRIPT_DETECTOR
            .detect_language_of(
                script_text
            )
        )

    elif script_name == "cyrillic":

        language = (
            CYRILLIC_SCRIPT_DETECTOR
            .detect_language_of(
                script_text
            )
        )

    else:

        language = (
            GENERAL_DETECTOR
            .detect_language_of(
                script_text
            )
        )

    if language is None:
        return None

    return language.iso_code_639_1.name.lower()


def detect_language(
    text: str,
) -> str | None:

    text = text.strip()

    if not text:
        return None

    counts = _count_script_letters(
        text
    )

    total_letters = sum(
        counts.values()
    )

    if total_letters == 0:
        return None

    # Check Arabic-script content independently from
    # Latin technical terms such as Network, Server, SLA, etc.
    arabic_words = _count_script_words(
        text,
        "arabic",
    )

    arabic_word_ratio = (
        arabic_words
        / max(
            len(
                text.split()
            ),
            1,
        )
    )

    if (
        counts["arabic"] >= MIN_SCRIPT_LETTERS
        and arabic_words >= MIN_SCRIPT_WORDS
        and arabic_word_ratio >= MIN_SCRIPT_WORD_RATIO
    ):

        language = _detect_from_script(
            text,
            "arabic",
        )

        if language is not None:
            return language

    # Apply the same principle to Cyrillic text
    # that may contain English technical terms.
    cyrillic_words = _count_script_words(
        text,
        "cyrillic",
    )

    cyrillic_word_ratio = (
        cyrillic_words
        / max(
            len(
                text.split()
            ),
            1,
        )
    )

    if (
        counts["cyrillic"] >= MIN_SCRIPT_LETTERS
        and cyrillic_words >= MIN_SCRIPT_WORDS
        and cyrillic_word_ratio >= MIN_SCRIPT_WORD_RATIO
    ):

        language = _detect_from_script(
            text,
            "cyrillic",
        )

        if language is not None:
            return language

    # Fall back to general detection for normal
    # single-language or mostly Latin documents.
    language = (
        GENERAL_DETECTOR
        .detect_language_of(text)
    )

    if language is None:
        return None

    return language.iso_code_639_1.name.lower()