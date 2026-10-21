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
    """Validate content supplied for analysis or document ingestion."""

    model_config = ConfigDict(
        extra="forbid",
    )

    text: str | None = Field(
        default=None,
        min_length=1,
        description="Text to analyze or index.",
    )

    url: HttpUrl | None = Field(
        default=None,
        description="URL of the content to process.",
    )

    source: str | None = Field(
        default=None,
        max_length=500,
        description=(
            "Optional source label for text input. "
            "URL inputs use the URL as their source."
        ),
    )

    task: TaskName | None = Field(
        default=None,
        description="Task to perform on the content.",
    )

    instruction: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Optional user instruction that can refine the task "
            "and determine the response language."
        ),
    )

    @field_validator("text")
    @classmethod
    def validate_text(
        cls,
        value: str | None,
    ) -> str | None:
        """Trim text and reject values containing only whitespace."""

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
    ) -> str | None:
        """Trim the instruction and reject blank values."""

        if value is not None:
            value = value.strip()

            if not value:
                raise ValueError(
                    "Instruction cannot contain only whitespace."
                )

        return value

    @field_validator("source")
    @classmethod
    def validate_source_label(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize an optional source label and reject blank labels."""

        if value is not None:
            value = value.strip()

            if not value:
                raise ValueError(
                    "Source cannot contain only whitespace."
                )

        return value

    @model_validator(mode="after")
    def validate_source(self) -> "ContentRequest":
        """Ensure exactly one content input is provided with valid metadata."""

        if self.text is None and self.url is None:
            raise ValueError(
                "Either text or url must be provided."
            )

        if self.text is not None and self.url is not None:
            raise ValueError(
                "Provide either text or url, not both."
            )

        if self.url is not None and self.source is not None:
            raise ValueError(
                "A custom source label can only be provided with text."
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

class RAGQueryRequest(BaseModel):
    """Validate a question and its retrieval limit for the RAG API."""

    model_config = ConfigDict(
        extra="forbid",
    )

    question: str = Field(
        min_length=1,
        max_length=8000,
        description="Question to answer using indexed content.",
    )

    k: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Maximum number of relevant document chunks to retrieve.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Remove surrounding whitespace and reject blank questions."""

        value = value.strip()

        if not value:
            raise ValueError(
                "Question cannot contain only whitespace."
            )

        return value

class DocumentIngestionResponse(BaseModel):
    """Describe the result of storing content in the vector database."""

    source_type: Literal["text", "url", "pdf"]

    source: str | None = None

    chunks_created: int = Field(
        ge=1,
        description="Number of document chunks stored.",
    )

    document_ids: list[str] = Field(
        min_length=1,
        description="IDs assigned to the stored document chunks.",
    )

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