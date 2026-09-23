from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


CategoryName = Literal[
    "technology",
    "economy",
    "politics",
    "science",
    "health",
    "culture",
    "sports",
    "media",
    "other",
]


TaskName = Literal[
    "summarize",
    "analyze",
    "ideas",
]


class ContentRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
    )

    text: str | None = Field(
        default=None,
        min_length=1,
        description="Text to analyze",
    )

    url: HttpUrl | None = Field(
        default=None,
        description="URL of the content to analyze",
    )

    task: TaskName | None = Field(
        default=None,
        description="Task to perform on the content",
    )

    instruction: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional user instruction that can refine the task "
            "and determine the response language"
        ),
    )

    @field_validator("text")
    @classmethod
    def validate_text(
        cls,
        value: str | None,
    ):
        if value is not None:
            value = value.strip()

            if not value:
                raise ValueError(
                    "Text cannot contain only whitespace."
                )

        return value

    @field_validator("instruction")
    @classmethod
    def validate_instruction(
        cls,
        value: str | None,
    ):
        if value is not None:
            value = value.strip()

            if not value:
                raise ValueError(
                    "Instruction cannot contain only whitespace."
                )

        return value

    @model_validator(mode="after")
    def validate_source(self):
        if (
            self.text is None
            and self.url is None
        ):
            raise ValueError(
                "Either text or url must be provided."
            )

        if (
            self.text is not None
            and self.url is not None
        ):
            raise ValueError(
                "Provide either text or url, not both."
            )

        return self


class NormalizedContent(BaseModel):
    content: str

    source_type: Literal[
        "text",
        "url",
        "pdf",
    ]

    source: str | None = None

    task: TaskName | None = None

    instruction: str | None = None

    source_language: str | None = None

class SummaryResult(BaseModel):
    title: str

    summary: str

    importance: int = Field(
        ge=1,
        le=100,
    )

    category: CategoryName


class AnalysisResult(BaseModel):
    title: str

    analysis: str

    importance: int = Field(
        ge=1,
        le=100,
    )

    category: CategoryName


class IdeasResult(BaseModel):
    title: str

    summary: str

    ideas: list[str] = Field(
        min_length=3,
        max_length=7,
    )

    category: CategoryName


class BaseContentResponse(BaseModel):
    input_type: Literal[
        "text",
        "url",
        "pdf",
    ]

    title: str

    category: CategoryName


class SummaryContentResponse(BaseContentResponse):
    summary: str

    importance: int = Field(
        ge=1,
        le=100,
    )


class AnalysisContentResponse(BaseContentResponse):
    analysis: str

    importance: int = Field(
        ge=1,
        le=100,
    )


class IdeasContentResponse(BaseContentResponse):
    summary: str

    ideas: list[str] = Field(
        min_length=3,
        max_length=7,
    )