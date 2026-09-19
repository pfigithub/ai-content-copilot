from typing import Literal

from pydantic import BaseModel, Field


class ContentRequest(BaseModel):
    text: str = Field(
        min_length=1,
        description="Text to analyze"
    )


class ContentResponse(BaseModel):
    input_type: Literal["text"]
    title: str
    summary: str
    importance: int = Field(
        ge=1,
        le=100
    )
    category: str