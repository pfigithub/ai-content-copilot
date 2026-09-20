from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


class ContentRequest(BaseModel):
    text: str | None = Field(
        default=None,
        min_length=1,
        description="Text to analyze",
    )

    url: HttpUrl | None = Field(
        default=None,
        description="URL of the content to analyze",
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str | None):
        if value is not None:
            value = value.strip()

            if not value:
                raise ValueError(
                    "Text cannot contain only whitespace."
                )

        return value

    @model_validator(mode="after")
    def validate_source(self):
        if self.text is None and self.url is None:
            raise ValueError(
                "Either text or url must be provided."
            )

        if self.text is not None and self.url is not None:
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


class AIContentResult(BaseModel):
    title: str
    summary: str
    importance: int = Field(
        ge=1,
        le=100,
    )
    category: str


class ContentResponse(BaseModel):
    input_type: Literal[
        "text",
        "url",
        "pdf",
    ]

    title: str
    summary: str

    importance: int = Field(
        ge=1,
        le=100,
    )

    category: str