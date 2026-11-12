from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.srs import UserQuestionSRS


def compute_sm2_next_state(
    repetitions: int,
    interval_days: int,
    ease_factor: float,
    is_correct: bool,
    now: datetime | None = None,
) -> tuple[int, int, float, datetime]:
    """Compute next SM-2 spaced repetition state.

    Maps binary quiz correctness to SM-2 quality:
    - correct -> quality = 4 (good recall)
    - incorrect -> quality = 1 (reset repetitions, review next day)
    """
    ref_now = now or datetime.now(UTC)
    quality = 4 if is_correct else 1

    if quality < 3:
        new_reps = 0
        new_interval = 1
    else:
        if repetitions == 0:
            new_interval = 1
        elif repetitions == 1:
            new_interval = 6
        else:
            new_interval = max(1, round(interval_days * ease_factor))
        new_reps = repetitions + 1

    new_ef = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    new_ef = max(1.3, round(new_ef, 4))
    next_review_at = ref_now + timedelta(days=new_interval)
    return new_reps, new_interval, new_ef, next_review_at


async def record_srs_answer(
    db: AsyncSession,
    user_id: UUID,
    question_id: UUID,
    is_correct: bool,
    now: datetime | None = None,
) -> UserQuestionSRS:
    ref_now = now or datetime.now(UTC)
    srs = await db.scalar(
        select(UserQuestionSRS).where(
            UserQuestionSRS.user_id == user_id,
            UserQuestionSRS.question_id == question_id,
        )
    )
    if srs is None:
        srs = UserQuestionSRS(
            user_id=user_id,
            question_id=question_id,
            repetitions=0,
            interval_days=0,
            ease_factor=2.5,
            total_attempts=0,
            correct_attempts=0,
        )
        db.add(srs)

    new_reps, new_interval, new_ef, next_review = compute_sm2_next_state(
        repetitions=srs.repetitions,
        interval_days=srs.interval_days,
        ease_factor=srs.ease_factor,
        is_correct=is_correct,
        now=ref_now,
    )
    srs.repetitions = new_reps
    srs.interval_days = new_interval
    srs.ease_factor = new_ef
    srs.next_review_at = next_review
    srs.last_reviewed_at = ref_now
    srs.total_attempts += 1
    if is_correct:
        srs.correct_attempts += 1
    return srs
