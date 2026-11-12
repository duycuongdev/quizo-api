import random
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.v1.deps import get_current_user, get_optional_current_user
from app.core.database import get_db
from app.models.quiz import (
    AttemptAnswer,
    Category,
    Question,
    Quiz,
    QuizAttempt,
    QuizQuestion,
)
from app.models.user import User
from app.schemas.quiz import (
    QuestionOptionResponse,
    QuestionResultDetail,
    QuizAttemptHistoryItem,
    QuizAttemptHistoryResponse,
    QuizCreateRequest,
    QuizOptionTakeSchema,
    QuizQuestionTakeSchema,
    QuizResultResponse,
    QuizSubmitRequest,
    QuizSummary,
    QuizTakeResponse,
)
from app.services.srs_service import record_srs_answer

router = APIRouter(prefix="/quizzes", tags=["Quizzes"])


def _to_quiz_summary(quiz: Quiz, current_user: User | None) -> QuizSummary:
    return QuizSummary(
        id=quiz.id,
        title=quiz.title,
        description=quiz.description,
        category_id=quiz.category_id,
        category_slug=quiz.category.slug if quiz.category else None,
        category_name=quiz.category.name if quiz.category else None,
        difficulty=quiz.difficulty,
        time_limit_seconds=quiz.time_limit_seconds,
        question_count=len(quiz.quiz_questions),
        is_public=quiz.is_public,
        is_owner=bool(current_user and quiz.user_id == current_user.id),
        created_at=quiz.created_at,
    )


def _to_quiz_take_response(quiz: Quiz) -> QuizTakeResponse:
    ordered_qq = sorted(quiz.quiz_questions, key=lambda x: x.order_index)
    questions_out = [
        QuizQuestionTakeSchema(
            id=qq.question.id,
            content=qq.question.content,
            code_snippet=qq.question.code_snippet,
            code_language=qq.question.code_language,
            difficulty=qq.question.difficulty,
            category_id=qq.question.category_id,
            order_index=qq.order_index,
            options=[
                QuizOptionTakeSchema(
                    id=opt.id,
                    content=opt.content,
                    order_index=opt.order_index,
                )
                for opt in sorted(qq.question.options, key=lambda o: o.order_index)
            ],
        )
        for qq in ordered_qq
    ]
    return QuizTakeResponse(
        id=quiz.id,
        title=quiz.title,
        description=quiz.description,
        category_id=quiz.category_id,
        category_slug=quiz.category.slug if quiz.category else None,
        category_name=quiz.category.name if quiz.category else None,
        difficulty=quiz.difficulty,
        time_limit_seconds=quiz.time_limit_seconds,
        question_count=len(questions_out),
        is_public=quiz.is_public,
        created_at=quiz.created_at,
        questions=questions_out,
    )


@router.get("", response_model=list[QuizSummary])
async def list_quizzes(
    category_slug: str | None = None,
    difficulty: str | None = Query(default=None, pattern="^(easy|medium|hard|mixed)$"),
    mine_only: bool = False,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=100),
    current_user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[QuizSummary]:
    if mine_only and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )

    conditions = []
    if mine_only and current_user is not None:
        conditions.append(Quiz.user_id == current_user.id)
    elif current_user is not None:
        conditions.append(or_(Quiz.is_public.is_(True), Quiz.user_id == current_user.id))
    else:
        conditions.append(Quiz.is_public.is_(True))

    if category_slug:
        cat = await db.scalar(select(Category).where(Category.slug == category_slug))
        if cat is None:
            return []
        conditions.append(Quiz.category_id == cat.id)
    if difficulty:
        conditions.append(Quiz.difficulty == difficulty)

    quizzes = (
        await db.scalars(
            select(Quiz)
            .options(selectinload(Quiz.category), selectinload(Quiz.quiz_questions))
            .where(*conditions)
            .order_by(Quiz.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
    ).all()
    return [_to_quiz_summary(q, current_user) for q in quizzes]


@router.post("", response_model=QuizTakeResponse, status_code=status.HTTP_201_CREATED)
async def create_quiz(
    payload: QuizCreateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuizTakeResponse:
    resolved_category: Category | None = None
    if payload.category_id is not None:
        resolved_category = await db.get(Category, payload.category_id)
        if resolved_category is None:
            raise HTTPException(status_code=404, detail="Category not found.")
    elif payload.category_slug:
        resolved_category = await db.scalar(
            select(Category).where(Category.slug == payload.category_slug)
        )
        if resolved_category is None:
            raise HTTPException(status_code=404, detail="Category not found.")

    selected_questions: list[Question] = []
    if payload.question_ids:
        unique_ids = list(dict.fromkeys(payload.question_ids))
        found = (
            await db.scalars(
                select(Question)
                .options(selectinload(Question.options))
                .where(Question.id.in_(unique_ids))
            )
        ).all()
        found_map = {q.id: q for q in found}
        for qid in unique_ids:
            q = found_map.get(qid)
            if q is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Question {qid} not found.",
                )
            if not q.is_public and q.user_id != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to include one or more selected questions.",
                )
            selected_questions.append(q)
    else:
        conditions = []
        if payload.include_mine:
            conditions.append(
                or_(Question.is_public.is_(True), Question.user_id == current_user.id)
            )
        else:
            conditions.append(Question.is_public.is_(True))
        if resolved_category is not None:
            conditions.append(Question.category_id == resolved_category.id)
        if payload.difficulty != "mixed":
            conditions.append(Question.difficulty == payload.difficulty)

        pool = (
            await db.scalars(
                select(Question)
                .options(selectinload(Question.options))
                .where(*conditions)
            )
        ).all()
        if not pool:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No matching questions available to create quiz.",
            )
        sample_size = min(payload.question_count, len(pool))
        selected_questions = random.sample(list(pool), sample_size)

    if not selected_questions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A quiz must contain at least one question.",
        )

    title = (payload.title or "").strip()
    if not title:
        cat_label = resolved_category.name if resolved_category else "AI Engineering"
        title = f"{cat_label} Practice Quiz"

    quiz = Quiz(
        user_id=current_user.id,
        category_id=resolved_category.id if resolved_category else selected_questions[0].category_id,
        title=title,
        description=payload.description,
        difficulty=payload.difficulty,
        time_limit_seconds=payload.time_limit_seconds,
        is_public=payload.is_public,
    )
    db.add(quiz)
    await db.flush()

    for idx, q in enumerate(selected_questions):
        db.add(QuizQuestion(quiz_id=quiz.id, question_id=q.id, order_index=idx))
    await db.commit()

    created = await db.scalar(
        select(Quiz)
        .options(
            selectinload(Quiz.category),
            selectinload(Quiz.quiz_questions)
            .selectinload(QuizQuestion.question)
            .selectinload(Question.options),
        )
        .where(Quiz.id == quiz.id)
    )
    assert created is not None
    return _to_quiz_take_response(created)


@router.get("/attempts/history", response_model=QuizAttemptHistoryResponse)
async def get_attempt_history(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=30, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuizAttemptHistoryResponse:
    total = await db.scalar(
        select(func.count(QuizAttempt.id)).where(QuizAttempt.user_id == current_user.id)
    )
    attempts = (
        await db.scalars(
            select(QuizAttempt)
            .options(selectinload(QuizAttempt.quiz).selectinload(Quiz.category))
            .where(QuizAttempt.user_id == current_user.id)
            .order_by(QuizAttempt.completed_at.desc())
            .offset(skip)
            .limit(limit)
        )
    ).all()

    items = [
        QuizAttemptHistoryItem(
            attempt_id=a.id,
            quiz_id=a.quiz_id,
            quiz_title=a.quiz.title,
            category_name=a.quiz.category.name if a.quiz.category else None,
            score=a.score,
            total_questions=a.total_questions,
            percentage=a.percentage,
            time_spent_seconds=a.time_spent_seconds,
            completed_at=a.completed_at,
        )
        for a in attempts
    ]
    return QuizAttemptHistoryResponse(items=items, total=int(total or 0))


@router.get("/attempts/{attempt_id}", response_model=QuizResultResponse)
async def get_attempt_detail(
    attempt_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuizResultResponse:
    attempt = await db.scalar(
        select(QuizAttempt)
        .options(
            selectinload(QuizAttempt.quiz),
            selectinload(QuizAttempt.answers)
            .selectinload(AttemptAnswer.question)
            .selectinload(Question.options),
        )
        .where(QuizAttempt.id == attempt_id)
    )
    if attempt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz attempt not found.",
        )
    if attempt.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this attempt.",
        )

    results: list[QuestionResultDetail] = []
    for ans in attempt.answers:
        q = ans.question
        ordered_opts = sorted(q.options, key=lambda o: o.order_index)
        correct_opt = next((o for o in ordered_opts if o.is_correct), None)
        results.append(
            QuestionResultDetail(
                question_id=q.id,
                content=q.content,
                code_snippet=q.code_snippet,
                code_language=q.code_language,
                selected_option_id=ans.selected_option_id,
                correct_option_id=correct_opt.id if correct_opt else None,
                is_correct=ans.is_correct,
                explanation=q.explanation,
                options=[QuestionOptionResponse.model_validate(o) for o in ordered_opts],
            )
        )

    return QuizResultResponse(
        attempt_id=attempt.id,
        quiz_id=attempt.quiz_id,
        quiz_title=attempt.quiz.title,
        score=attempt.score,
        total_questions=attempt.total_questions,
        percentage=attempt.percentage,
        time_spent_seconds=attempt.time_spent_seconds,
        completed_at=attempt.completed_at,
        results=results,
    )


@router.get("/{quiz_id}", response_model=QuizTakeResponse)
async def get_quiz(
    quiz_id: UUID,
    current_user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuizTakeResponse:
    quiz = await db.scalar(
        select(Quiz)
        .options(
            selectinload(Quiz.category),
            selectinload(Quiz.quiz_questions)
            .selectinload(QuizQuestion.question)
            .selectinload(Question.options),
        )
        .where(Quiz.id == quiz_id)
    )
    if quiz is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz not found.",
        )
    if not quiz.is_public:
        if current_user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required to access this quiz.",
            )
        if quiz.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this quiz.",
            )

    return _to_quiz_take_response(quiz)


@router.post("/{quiz_id}/submit", response_model=QuizResultResponse)
async def submit_quiz(
    quiz_id: UUID,
    payload: QuizSubmitRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuizResultResponse:
    quiz = await db.scalar(
        select(Quiz)
        .options(
            selectinload(Quiz.quiz_questions)
            .selectinload(QuizQuestion.question)
            .selectinload(Question.options),
        )
        .where(Quiz.id == quiz_id)
    )
    if quiz is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Quiz not found.",
        )
    if not quiz.is_public and quiz.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to submit this quiz.",
        )

    ordered_qq = sorted(quiz.quiz_questions, key=lambda x: x.order_index)
    quiz_q_map = {qq.question.id: qq.question for qq in ordered_qq}

    # Deduplicate submitted answers and validate option ownership
    submitted_map: dict[UUID, UUID | None] = {}
    for item in payload.answers:
        if item.question_id not in quiz_q_map:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Question {item.question_id} does not belong to this quiz.",
            )
        if item.selected_option_id is not None:
            valid_option_ids = {opt.id for opt in quiz_q_map[item.question_id].options}
            if item.selected_option_id not in valid_option_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Selected option does not belong to the specified question.",
                )
        submitted_map[item.question_id] = item.selected_option_id

    total_questions = len(ordered_qq)
    score = 0
    attempt = QuizAttempt(
        quiz_id=quiz.id,
        user_id=current_user.id,
        score=0,
        total_questions=total_questions,
        percentage=0.0,
        time_spent_seconds=payload.time_spent_seconds,
    )
    db.add(attempt)
    await db.flush()

    results: list[QuestionResultDetail] = []
    for qq in ordered_qq:
        q = qq.question
        ordered_opts = sorted(q.options, key=lambda o: o.order_index)
        correct_opt = next((o for o in ordered_opts if o.is_correct), None)
        selected_opt_id = submitted_map.get(q.id)
        is_correct = bool(correct_opt and selected_opt_id == correct_opt.id)
        if is_correct:
            score += 1

        db.add(
            AttemptAnswer(
                attempt_id=attempt.id,
                question_id=q.id,
                selected_option_id=selected_opt_id,
                is_correct=is_correct,
            )
        )
        # Update SM-2 spaced repetition state if user answered this question
        if q.id in submitted_map:
            await record_srs_answer(
                db=db,
                user_id=current_user.id,
                question_id=q.id,
                is_correct=is_correct,
            )

        results.append(
            QuestionResultDetail(
                question_id=q.id,
                content=q.content,
                code_snippet=q.code_snippet,
                code_language=q.code_language,
                selected_option_id=selected_opt_id,
                correct_option_id=correct_opt.id if correct_opt else None,
                is_correct=is_correct,
                explanation=q.explanation,
                options=[QuestionOptionResponse.model_validate(o) for o in ordered_opts],
            )
        )

    percentage = round((score / total_questions) * 100, 2) if total_questions > 0 else 0.0
    attempt.score = score
    attempt.percentage = percentage
    await db.commit()
    await db.refresh(attempt)

    return QuizResultResponse(
        attempt_id=attempt.id,
        quiz_id=quiz.id,
        quiz_title=quiz.title,
        score=score,
        total_questions=total_questions,
        percentage=percentage,
        time_spent_seconds=attempt.time_spent_seconds,
        completed_at=attempt.completed_at,
        results=results,
    )
