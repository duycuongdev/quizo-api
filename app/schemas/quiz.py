from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class QuestionOptionCreate(BaseModel):
    content: str = Field(min_length=1, max_length=1000)
    is_correct: bool = False
    order_index: int | None = None

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Option content cannot be empty.")
        return cleaned


class QuestionOptionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    content: str
    is_correct: bool
    order_index: int


class QuestionCreate(BaseModel):
    category_id: UUID | None = None
    category_slug: str | None = None
    content: str = Field(min_length=5, max_length=4000)
    code_snippet: str | None = Field(default=None, max_length=8000)
    code_language: str | None = Field(default=None, max_length=40)
    explanation: str = Field(min_length=1, max_length=4000)
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    source: Literal["manual", "bulk", "ai"] = "manual"
    is_public: bool = False
    document_id: UUID | None = None
    options: list[QuestionOptionCreate] = Field(min_length=2, max_length=6)

    @field_validator("content", "explanation")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Field cannot be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_question(self) -> "QuestionCreate":
        if self.category_id is None and not self.category_slug:
            raise ValueError("Either category_id or category_slug is required.")
        correct_count = sum(1 for opt in self.options if opt.is_correct)
        if correct_count != 1:
            raise ValueError("Exactly one option must be marked as correct.")
        return self


class QuestionUpdate(BaseModel):
    category_id: UUID | None = None
    category_slug: str | None = None
    content: str | None = Field(default=None, min_length=5, max_length=4000)
    code_snippet: str | None = Field(default=None, max_length=8000)
    code_language: str | None = Field(default=None, max_length=40)
    explanation: str | None = Field(default=None, min_length=1, max_length=4000)
    difficulty: Literal["easy", "medium", "hard"] | None = None
    is_public: bool | None = None
    options: list[QuestionOptionCreate] | None = Field(default=None, min_length=2, max_length=6)

    @model_validator(mode="after")
    def validate_options(self) -> "QuestionUpdate":
        if self.options is not None:
            correct_count = sum(1 for opt in self.options if opt.is_correct)
            if correct_count != 1:
                raise ValueError("Exactly one option must be marked as correct.")
        return self


class QuestionBulkCreate(BaseModel):
    questions: list[QuestionCreate] = Field(min_length=1, max_length=100)


class QuestionResponse(BaseModel):
    id: UUID
    category_id: UUID
    category_slug: str
    category_name: str
    user_id: UUID | None
    is_owner: bool
    document_id: UUID | None = None
    content: str
    code_snippet: str | None = None
    code_language: str | None = None
    explanation: str
    difficulty: str
    source: str
    is_public: bool
    options: list[QuestionOptionResponse]
    created_at: datetime


class QuestionListResponse(BaseModel):
    items: list[QuestionResponse]
    total: int
    skip: int
    limit: int


class QuestionBulkResponse(BaseModel):
    created_count: int
    questions: list[QuestionResponse]


class QuizOptionTakeSchema(BaseModel):
    """Quiz-taking option schema. Intentionally omits is_correct for anti-cheat."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    content: str
    order_index: int


class QuizQuestionTakeSchema(BaseModel):
    """Quiz-taking question schema. Intentionally omits explanation for anti-cheat."""

    id: UUID
    content: str
    code_snippet: str | None = None
    code_language: str | None = None
    difficulty: str
    category_id: UUID
    order_index: int
    options: list[QuizOptionTakeSchema]


class QuizSummary(BaseModel):
    id: UUID
    title: str
    description: str | None = None
    category_id: UUID | None = None
    category_slug: str | None = None
    category_name: str | None = None
    difficulty: str
    time_limit_seconds: int | None = None
    question_count: int
    is_public: bool
    is_owner: bool
    created_at: datetime


class QuizTakeResponse(BaseModel):
    id: UUID
    title: str
    description: str | None = None
    category_id: UUID | None = None
    category_slug: str | None = None
    category_name: str | None = None
    difficulty: str
    time_limit_seconds: int | None = None
    question_count: int
    is_public: bool
    created_at: datetime
    questions: list[QuizQuestionTakeSchema]


class QuizCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=255)
    description: str | None = Field(default=None, max_length=1000)
    category_id: UUID | None = None
    category_slug: str | None = None
    difficulty: Literal["easy", "medium", "hard", "mixed"] = "mixed"
    question_count: int = Field(default=10, ge=1, le=50)
    question_ids: list[UUID] | None = Field(default=None, max_length=50)
    time_limit_seconds: int | None = Field(default=None, ge=30, le=7200)
    include_mine: bool = True
    is_public: bool = False


class SubmitAnswerItem(BaseModel):
    question_id: UUID
    selected_option_id: UUID | None = None


class QuizSubmitRequest(BaseModel):
    answers: list[SubmitAnswerItem] = Field(min_length=1, max_length=100)
    time_spent_seconds: int = Field(default=0, ge=0, le=86400)


class QuestionResultDetail(BaseModel):
    question_id: UUID
    content: str
    code_snippet: str | None = None
    code_language: str | None = None
    selected_option_id: UUID | None = None
    correct_option_id: UUID | None = None
    is_correct: bool
    explanation: str
    options: list[QuestionOptionResponse]


class QuizResultResponse(BaseModel):
    attempt_id: UUID
    quiz_id: UUID
    quiz_title: str
    score: int
    total_questions: int
    percentage: float
    time_spent_seconds: int
    completed_at: datetime
    results: list[QuestionResultDetail]


class QuizAttemptHistoryItem(BaseModel):
    attempt_id: UUID
    quiz_id: UUID
    quiz_title: str
    category_name: str | None = None
    score: int
    total_questions: int
    percentage: float
    time_spent_seconds: int
    completed_at: datetime


class QuizAttemptHistoryResponse(BaseModel):
    items: list[QuizAttemptHistoryItem]
    total: int


class CategoryCreateRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: str | None = Field(default=None, max_length=500)
    icon: str = Field(default="folder", max_length=50)
    color: str = Field(default="#4F46E5", max_length=30)


class CategoryResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    description: str
    icon: str
    color: str
    order_index: int
    question_count: int


class CategoryDetailResponse(CategoryResponse):
    difficulty_counts: dict[str, int]
    quizzes: list[QuizSummary]
