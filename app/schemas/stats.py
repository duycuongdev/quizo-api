from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.schemas.quiz import QuestionOptionResponse, QuizOptionTakeSchema


class CategoryAccuracyItem(BaseModel):
    category_id: UUID
    category_slug: str
    category_name: str
    attempted: int
    correct: int
    accuracy: float


class AccuracyTrendItem(BaseModel):
    date: str
    attempts: int
    questions_answered: int
    accuracy: float


class DailyActivityItem(BaseModel):
    date: str
    count: int


class StatsOverviewResponse(BaseModel):
    total_attempts: int
    total_questions_answered: int
    correct_answers: int
    overall_accuracy: float
    current_streak_days: int
    total_time_spent_seconds: int
    due_srs_count: int
    category_breakdown: list[CategoryAccuracyItem]
    accuracy_trend: list[AccuracyTrendItem]
    daily_activity: list[DailyActivityItem]


class SrsDueQuestionItem(BaseModel):
    srs_id: UUID
    question_id: UUID
    category_id: UUID
    category_slug: str
    category_name: str
    content: str
    code_snippet: str | None = None
    code_language: str | None = None
    difficulty: str
    repetitions: int
    interval_days: int
    ease_factor: float
    next_review_at: datetime
    options: list[QuizOptionTakeSchema]


class SrsDueResponse(BaseModel):
    due_count: int
    items: list[SrsDueQuestionItem]


class SrsReviewRequest(BaseModel):
    question_id: UUID
    selected_option_id: UUID


class SrsReviewResponse(BaseModel):
    question_id: UUID
    selected_option_id: UUID
    correct_option_id: UUID | None
    is_correct: bool
    explanation: str
    repetitions: int
    interval_days: int
    ease_factor: float
    next_review_at: datetime
    options: list[QuestionOptionResponse]
