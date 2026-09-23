import pytest

from app.services.output_language_service import (
    resolve_output_language,
)


@pytest.mark.parametrize(
    "instruction, source_language, expected",
    [
        (
            "Answer in English.",
            "fa",
            "en",
        ),
        (
            "این متن را به انگلیسی خلاصه کن.",
            "fa",
            "en",
        ),
        (
            "این متن انگلیسی را به فارسی خلاصه کن.",
            "en",
            "fa",
        ),
        (
            "Please summarize this in Persian.",
            "en",
            "fa",
        ),
        (
            "Write the summary in Spanish.",
            "en",
            "es",
        ),
        (
            "Ответь по-английски.",
            "ru",
            "en",
        ),
        (
            "روی پیامدهای اقتصادی تمرکز کن.",
            "fa",
            "fa",
        ),
        (
            None,
            "fa",
            "fa",
        ),
    ],
)
def test_resolve_output_language(
    instruction,
    source_language,
    expected,
):

    assert (
        resolve_output_language(
            instruction,
            source_language,
        )
        == expected
    )