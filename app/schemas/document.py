from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class DocumentTextCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    content_text: str = Field(min_length=50, max_length=150000)

    @field_validator("title", "content_text")
    @classmethod
    def clean_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty.")
        return cleaned


class DocumentResponse(BaseModel):
    id: UUID
    title: str
    original_filename: str | None = None
    file_type: str
    file_size_bytes: int
    char_count: int
    preview: str
    created_at: datetime


class DocumentGenerateRequest(BaseModel):
    num_questions: int = Field(default=5, ge=3, le=15)
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "mixed"
    language: Literal["vi", "en"] = "vi"
    category_slug: str | None = None
    category_id: UUID | None = None
    save_to_bank: bool = False


class GeneratedOptionItem(BaseModel):
    content: str
    is_correct: bool


class GeneratedQuestionItem(BaseModel):
    id: UUID | None = None
    category_id: UUID | None = None
    category_slug: str | None = None
    content: str
    code_snippet: str | None = None
    code_language: str | None = None
    explanation: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    options: list[GeneratedOptionItem]


class DocumentGenerateResponse(BaseModel):
    document_id: UUID
    category_id: UUID | None = None
    category_slug: str | None = None
    saved: bool
    questions: list[GeneratedQuestionItem]
