from collections import defaultdict
from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.v1.deps import get_current_user
from app.core.database import get_db
from app.models.quiz import AttemptAnswer, Category, Question, QuizAttempt
from app.models.srs import UserQuestionSRS
from app.models.user import User
from app.schemas.quiz import QuestionOptionResponse, QuizOptionTakeSchema
from app.schemas.stats import (
    AccuracyTrendItem,
    CategoryAccuracyItem,
    DailyActivityItem,
    SrsDueQuestionItem,
    SrsDueResponse,
    SrsReviewRequest,
    SrsReviewResponse,
    StatsOverviewResponse,
)
from app.services.srs_service import record_srs_answer

router = APIRouter(prefix="/stats", tags=["Stats & SRS"])


def _to_utc_date(dt: datetime) -> date:
    if dt.tzinfo is None:
        return dt.date()
    return dt.astimezone(UTC).date()


def _compute_streak(active_dates: set[date], today: date) -> int:
    if not active_dates:
        return 0
    cursor = today if today in active_dates else (today - timedelta(days=1))
    if cursor not in active_dates:
        return 0
    streak = 0
    while cursor in active_dates:
        streak += 1
        cursor -= timedelta(days=1)
    return streak


@router.get("/overview", response_model=StatsOverviewResponse)
async def get_stats_overview(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> StatsOverviewResponse:
    now = datetime.now(UTC)
    today = now.date()

    attempts = (
        await db.scalars(
            select(QuizAttempt)
            .where(QuizAttempt.user_id == current_user.id)
            .order_by(QuizAttempt.completed_at.asc())
        )
    ).all()

    total_attempts = len(attempts)
    total_time_spent = sum(a.time_spent_seconds for a in attempts)

    answers = (
        await db.scalars(
            select(AttemptAnswer)
            .join(QuizAttempt, AttemptAnswer.attempt_id == QuizAttempt.id)
            .options(selectinload(AttemptAnswer.question).selectinload(Question.category))
            .where(QuizAttempt.user_id == current_user.id)
        )
    ).all()

    total_questions_answered = len(answers)
    correct_answers = sum(1 for a in answers if a.is_correct)
    overall_accuracy = (
        round((correct_answers / total_questions_answered) * 100, 2)
        if total_questions_answered > 0
        else 0.0
    )

    # Category accuracy breakdown across all categories
    categories = (
        await db.scalars(select(Category).order_by(Category.order_index))
    ).all()
    cat_stats: dict = {cat.id: [0, 0] for cat in categories}
    for ans in answers:
        cid = ans.question.category_id
        if cid in cat_stats:
            cat_stats[cid][0] += 1
            if ans.is_correct:
                cat_stats[cid][1] += 1

    category_breakdown = [
        CategoryAccuracyItem(
            category_id=cat.id,
            category_slug=cat.slug,
            category_name=cat.name,
            attempted=cat_stats[cat.id][0],
            correct=cat_stats[cat.id][1],
            accuracy=round((cat_stats[cat.id][1] / cat_stats[cat.id][0]) * 100, 2)
            if cat_stats[cat.id][0] > 0
            else 0.0,
        )
        for cat in categories
    ]

    # Trend & daily activity
    day_attempts: dict[date, list[QuizAttempt]] = defaultdict(list)
    for a in attempts:
        day_attempts[_to_utc_date(a.completed_at)].append(a)

    active_dates = set(day_attempts.keys())
    current_streak = _compute_streak(active_dates, today)

    start_day = today - timedelta(days=29)
    daily_activity = []
    for offset in range(30):
        d = start_day + timedelta(days=offset)
        daily_activity.append(
            DailyActivityItem(date=d.isoformat(), count=len(day_attempts.get(d, [])))
        )

    accuracy_trend = []
    for d in sorted(active_dates)[-14:]:
        day_list = day_attempts[d]
        q_sum = sum(x.total_questions for x in day_list)
        s_sum = sum(x.score for x in day_list)
        acc = round((s_sum / q_sum) * 100, 2) if q_sum > 0 else 0.0
        accuracy_trend.append(
            AccuracyTrendItem(
                date=d.isoformat(),
                attempts=len(day_list),
                questions_answered=q_sum,
                accuracy=acc,
            )
        )

    due_srs_count = await db.scalar(
        select(func.count(UserQuestionSRS.id)).where(
            UserQuestionSRS.user_id == current_user.id,
            UserQuestionSRS.next_review_at <= now,
        )
    )

    return StatsOverviewResponse(
        total_attempts=total_attempts,
        total_questions_answered=total_questions_answered,
        correct_answers=correct_answers,
        overall_accuracy=overall_accuracy,
        current_streak_days=current_streak,
        total_time_spent_seconds=total_time_spent,
        due_srs_count=int(due_srs_count or 0),
        category_breakdown=category_breakdown,
        accuracy_trend=accuracy_trend,
        daily_activity=daily_activity,
    )


@router.get("/srs/due", response_model=SrsDueResponse)
async def get_due_srs_questions(
    category_slug: str | None = None,
    limit: int = Query(default=20, ge=1, le=50),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SrsDueResponse:
    now = datetime.now(UTC)
    stmt = (
        select(UserQuestionSRS, Question)
        .join(Question, UserQuestionSRS.question_id == Question.id)
        .options(selectinload(Question.options), selectinload(Question.category))
        .where(
            UserQuestionSRS.user_id == current_user.id,
            UserQuestionSRS.next_review_at <= now,
        )
        .order_by(UserQuestionSRS.next_review_at.asc())
    )

    if category_slug:
        cat = await db.scalar(select(Category).where(Category.slug == category_slug))
        if cat is None:
            return SrsDueResponse(due_count=0, items=[])
        stmt = stmt.where(Question.category_id == cat.id)

    rows = (await db.execute(stmt)).all()
    due_count = len(rows)
    items: list[SrsDueQuestionItem] = []
    for srs, q in rows[:limit]:
        items.append(
            SrsDueQuestionItem(
                srs_id=srs.id,
                question_id=q.id,
                category_id=q.category_id,
                category_slug=q.category.slug,
                category_name=q.category.name,
                content=q.content,
                code_snippet=q.code_snippet,
                code_language=q.code_language,
                difficulty=q.difficulty,
                repetitions=srs.repetitions,
                interval_days=srs.interval_days,
                ease_factor=srs.ease_factor,
                next_review_at=srs.next_review_at,
                options=[
                    QuizOptionTakeSchema(
                        id=opt.id,
                        content=opt.content,
                        order_index=opt.order_index,
                    )
                    for opt in sorted(q.options, key=lambda o: o.order_index)
                ],
            )
        )

    return SrsDueResponse(due_count=due_count, items=items)


@router.post("/srs/review", response_model=SrsReviewResponse)
async def review_srs_question(
    payload: SrsReviewRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SrsReviewResponse:
    question = await db.scalar(
        select(Question)
        .options(selectinload(Question.options))
        .where(Question.id == payload.question_id)
    )
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found.",
        )
    if not question.is_public and question.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to review this question.",
        )

    ordered_opts = sorted(question.options, key=lambda o: o.order_index)
    if payload.selected_option_id not in {o.id for o in ordered_opts}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Selected option does not belong to this question.",
        )

    correct_opt = next((o for o in ordered_opts if o.is_correct), None)
    is_correct = bool(correct_opt and payload.selected_option_id == correct_opt.id)

    srs = await record_srs_answer(
        db=db,
        user_id=current_user.id,
        question_id=question.id,
        is_correct=is_correct,
    )
    await db.commit()
    await db.refresh(srs)

    return SrsReviewResponse(
        question_id=question.id,
        selected_option_id=payload.selected_option_id,
        correct_option_id=correct_opt.id if correct_opt else None,
        is_correct=is_correct,
        explanation=question.explanation,
        repetitions=srs.repetitions,
        interval_days=srs.interval_days,
        ease_factor=srs.ease_factor,
        next_review_at=srs.next_review_at,
        options=[QuestionOptionResponse.model_validate(o) for o in ordered_opts],
    )
