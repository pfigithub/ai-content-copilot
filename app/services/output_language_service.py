import re
import unicodedata
from functools import lru_cache

from langcodes import find


LANGUAGE_ALIASES = {
    "english": "en",
    "persian": "fa",
    "farsi": "fa",
    "russian": "ru",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "arabic": "ar",
    "turkish": "tr",

    "انگلیسی": "en",
    "فارسی": "fa",
    "روسی": "ru",
    "اسپانیایی": "es",
    "فرانسوی": "fr",
    "آلمانی": "de",
    "عربی": "ar",
    "ترکی": "tr",

    "английский": "en",
    "английском": "en",
    "английски": "en",
    "по-английски": "en",
    "персидский": "fa",
    "персидском": "fa",
    "русский": "ru",
    "русском": "ru",
    "по-русски": "ru",
}


LANGUAGE_MARKERS = (
    "in",
    "into",
    "to",
    "به",
    "در",
    "на",
    "по",
)


LEADING_LANGUAGE_WORDS = {
    "language",
    "زبان",
    "زبانِ",
}


def _normalize_text(
    text: str,
) -> str:

    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    text = (
        text.replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
    )

    return text.casefold().strip()


@lru_cache(maxsize=256)
def _resolve_language_name(
    name: str,
) -> str | None:

    normalized_name = _normalize_text(
        name
    )

    if not normalized_name:
        return None

    alias = LANGUAGE_ALIASES.get(
        normalized_name
    )

    if alias is not None:
        return alias

    try:

        language = find(
            normalized_name
        )

    except LookupError:
        return None

    return language.language or None


def _resolve_candidate_tokens(
    tokens: list[str],
) -> str | None:

    if not tokens:
        return None

    tokens = list(tokens)

    while (
        tokens
        and tokens[0] in LEADING_LANGUAGE_WORDS
    ):
        tokens.pop(0)

    if not tokens:
        return None

    max_words = min(
        len(tokens),
        3,
    )

    for size in range(
        max_words,
        0,
        -1,
    ):

        candidate = " ".join(
            tokens[:size]
        )

        language = (
            _resolve_language_name(
                candidate
            )
        )

        if language is not None:
            return language

    return None


def resolve_output_language(
    instruction: str | None,
    source_language: str | None,
) -> str | None:

    if not instruction:
        return source_language

    normalized = _normalize_text(
        instruction
    )

    if not normalized:
        return source_language

    tokens = re.findall(
        r"[\w\u0600-\u06ff\u0400-\u052f-]+",
        normalized,
    )

    for index, token in enumerate(tokens):

        if token not in LANGUAGE_MARKERS:
            continue

        candidate_tokens = tokens[
            index + 1:
            index + 4
        ]

        language = (
            _resolve_candidate_tokens(
                candidate_tokens
            )
        )

        if language is not None:
            return language

    # Support instructions such as:
    # "زبان پاسخ انگلیسی باشد"
    # or "language English"
    for index, token in enumerate(tokens):

        if token not in (
            "language",
            "زبان",
        ):
            continue

        language = (
            _resolve_candidate_tokens(
                tokens[
                    index + 1:
                    index + 4
                ]
            )
        )

        if language is not None:
            return language

    # Direct multilingual forms such as:
    # "ответь по-английски"
    for token in tokens:

        language = (
            _resolve_language_name(
                token
            )
        )

        if language is not None:
            if token in LANGUAGE_ALIASES:
                return language

    return source_language