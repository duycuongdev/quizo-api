from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.v1.deps import get_optional_current_user
from app.core.database import get_db
from app.models.quiz import Category, Question, Quiz
from app.models.user import User
from app.schemas.quiz import CategoryDetailResponse, CategoryResponse, QuizSummary

router = APIRouter(prefix="/categories", tags=["Categories"])


def _visibility_filter(current_user: User | None):
    if current_user is not None:
        return or_(Question.is_public.is_(True), Question.user_id == current_user.id)
    return Question.is_public.is_(True)


@router.get("", response_model=list[CategoryResponse])
async def list_categories(
    current_user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[CategoryResponse]:
    categories = (
        await db.scalars(select(Category).order_by(Category.order_index, Category.name))
    ).all()

    counts_stmt = (
        select(Question.category_id, func.count(Question.id))
        .where(_visibility_filter(current_user))
        .group_by(Question.category_id)
    )
    counts_rows = (await db.execute(counts_stmt)).all()
    counts_map = {row[0]: int(row[1]) for row in counts_rows}

    return [
        CategoryResponse(
            id=cat.id,
            name=cat.name,
            slug=cat.slug,
            description=cat.description,
            icon=cat.icon,
            color=cat.color,
            order_index=cat.order_index,
            question_count=counts_map.get(cat.id, 0),
        )
        for cat in categories
    ]


@router.get("/{slug}", response_model=CategoryDetailResponse)
async def get_category_detail(
    slug: str,
    current_user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
) -> CategoryDetailResponse:
    category = await db.scalar(select(Category).where(Category.slug == slug))
    if category is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found.",
        )

    diff_rows = (
        await db.execute(
            select(Question.difficulty, func.count(Question.id))
            .where(
                Question.category_id == category.id,
                _visibility_filter(current_user),
            )
            .group_by(Question.difficulty)
        )
    ).all()
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    total_q = 0
    for diff, cnt in diff_rows:
        difficulty_counts[diff] = int(cnt)
        total_q += int(cnt)

    quiz_filter = (
        or_(Quiz.is_public.is_(True), Quiz.user_id == current_user.id)
        if current_user is not None
        else Quiz.is_public.is_(True)
    )
    quizzes = (
        await db.scalars(
            select(Quiz)
            .options(selectinload(Quiz.quiz_questions))
            .where(Quiz.category_id == category.id, quiz_filter)
            .order_by(Quiz.created_at.desc())
        )
    ).all()

    quiz_summaries = [
        QuizSummary(
            id=q.id,
            title=q.title,
            description=q.description,
            category_id=q.category_id,
            category_slug=category.slug,
            category_name=category.name,
            difficulty=q.difficulty,
            time_limit_seconds=q.time_limit_seconds,
            question_count=len(q.quiz_questions),
            is_public=q.is_public,
            is_owner=bool(current_user and q.user_id == current_user.id),
            created_at=q.created_at,
        )
        for q in quizzes
    ]

    return CategoryDetailResponse(
        id=category.id,
        name=category.name,
        slug=category.slug,
        description=category.description,
        icon=category.icon,
        color=category.color,
        order_index=category.order_index,
        question_count=total_q,
        difficulty_counts=difficulty_counts,
        quizzes=quiz_summaries,
    )
