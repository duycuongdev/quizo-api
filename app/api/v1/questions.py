from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.v1.deps import get_current_user, get_optional_current_user
from app.core.database import get_db
from app.models.document import Document
from app.models.quiz import Category, Question, QuestionOption
from app.models.user import User
from app.schemas.quiz import (
    QuestionBulkCreate,
    QuestionBulkResponse,
    QuestionCreate,
    QuestionListResponse,
    QuestionOptionResponse,
    QuestionResponse,
    QuestionUpdate,
)

router = APIRouter(prefix="/questions", tags=["Questions"])


def _to_question_response(q: Question, current_user: User | None) -> QuestionResponse:
    return QuestionResponse(
        id=q.id,
        category_id=q.category_id,
        category_slug=q.category.slug,
        category_name=q.category.name,
        user_id=q.user_id,
        is_owner=bool(current_user and q.user_id == current_user.id),
        document_id=q.document_id,
        content=q.content,
        code_snippet=q.code_snippet,
        code_language=q.code_language,
        explanation=q.explanation,
        difficulty=q.difficulty,
        source=q.source,
        is_public=q.is_public,
        options=[
            QuestionOptionResponse.model_validate(opt)
            for opt in sorted(q.options, key=lambda o: o.order_index)
        ],
        created_at=q.created_at,
    )


async def _resolve_category(
    db: AsyncSession, category_id: UUID | None, category_slug: str | None
) -> Category:
    category: Category | None = None
    if category_id is not None:
        category = await db.get(Category, category_id)
    elif category_slug:
        category = await db.scalar(select(Category).where(Category.slug == category_slug))
    if category is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Category not found.",
        )
    return category


async def _validate_document_ownership(
    db: AsyncSession, document_id: UUID | None, user_id: UUID
) -> None:
    if document_id is None:
        return
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    if doc.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not own the referenced document.",
        )


@router.get("", response_model=QuestionListResponse)
async def list_questions(
    category_slug: str | None = None,
    category_id: UUID | None = None,
    difficulty: str | None = Query(default=None, pattern="^(easy|medium|hard)$"),
    source: str | None = Query(default=None, pattern="^(seed|manual|bulk|ai)$"),
    search: str | None = Query(default=None, max_length=200),
    mine_only: bool = False,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    current_user: User | None = Depends(get_optional_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuestionListResponse:
    if mine_only and current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to filter your own questions.",
        )

    conditions = []
    if mine_only and current_user is not None:
        conditions.append(Question.user_id == current_user.id)
    elif current_user is not None:
        conditions.append(
            or_(Question.is_public.is_(True), Question.user_id == current_user.id)
        )
    else:
        conditions.append(Question.is_public.is_(True))

    if category_id is not None:
        conditions.append(Question.category_id == category_id)
    if category_slug:
        cat = await db.scalar(select(Category).where(Category.slug == category_slug))
        if cat is None:
            return QuestionListResponse(items=[], total=0, skip=skip, limit=limit)
        conditions.append(Question.category_id == cat.id)
    if difficulty:
        conditions.append(Question.difficulty == difficulty)
    if source:
        conditions.append(Question.source == source)
    if search and search.strip():
        pattern = f"%{search.strip()}%"
        conditions.append(
            or_(
                Question.content.ilike(pattern),
                Question.explanation.ilike(pattern),
            )
        )

    total = await db.scalar(
        select(func.count(Question.id)).where(*conditions)
    )
    rows = (
        await db.scalars(
            select(Question)
            .options(selectinload(Question.options), selectinload(Question.category))
            .where(*conditions)
            .order_by(Question.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
    ).all()

    return QuestionListResponse(
        items=[_to_question_response(q, current_user) for q in rows],
        total=int(total or 0),
        skip=skip,
        limit=limit,
    )


@router.post("", response_model=QuestionResponse, status_code=status.HTTP_201_CREATED)
async def create_question(
    payload: QuestionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuestionResponse:
    category = await _resolve_category(db, payload.category_id, payload.category_slug)
    await _validate_document_ownership(db, payload.document_id, current_user.id)

    question = Question(
        category_id=category.id,
        user_id=current_user.id,
        document_id=payload.document_id,
        content=payload.content,
        code_snippet=payload.code_snippet,
        code_language=payload.code_language,
        explanation=payload.explanation,
        difficulty=payload.difficulty,
        source=payload.source,
        is_public=payload.is_public,
    )
    for idx, opt in enumerate(payload.options):
        question.options.append(
            QuestionOption(
                content=opt.content,
                is_correct=opt.is_correct,
                order_index=opt.order_index if opt.order_index is not None else idx,
            )
        )
    db.add(question)
    await db.commit()

    created = await db.scalar(
        select(Question)
        .options(selectinload(Question.options), selectinload(Question.category))
        .where(Question.id == question.id)
    )
    assert created is not None
    return _to_question_response(created, current_user)


@router.post("/bulk", response_model=QuestionBulkResponse, status_code=status.HTTP_201_CREATED)
async def bulk_create_questions(
    payload: QuestionBulkCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuestionBulkResponse:
    category_cache: dict[str, Category] = {}
    created_ids: list[UUID] = []

    for item in payload.questions:
        cache_key = str(item.category_id or item.category_slug)
        if cache_key not in category_cache:
            category_cache[cache_key] = await _resolve_category(
                db, item.category_id, item.category_slug
            )
        category = category_cache[cache_key]
        await _validate_document_ownership(db, item.document_id, current_user.id)

        source = item.source if item.source in {"ai", "bulk"} else "bulk"
        question = Question(
            category_id=category.id,
            user_id=current_user.id,
            document_id=item.document_id,
            content=item.content,
            code_snippet=item.code_snippet,
            code_language=item.code_language,
            explanation=item.explanation,
            difficulty=item.difficulty,
            source=source,
            is_public=item.is_public,
        )
        for idx, opt in enumerate(item.options):
            question.options.append(
                QuestionOption(
                    content=opt.content,
                    is_correct=opt.is_correct,
                    order_index=opt.order_index if opt.order_index is not None else idx,
                )
            )
        db.add(question)
        await db.flush()
        created_ids.append(question.id)

    await db.commit()

    rows = (
        await db.scalars(
            select(Question)
            .options(selectinload(Question.options), selectinload(Question.category))
            .where(Question.id.in_(created_ids))
        )
    ).all()
    row_map = {r.id: r for r in rows}
    ordered = [_to_question_response(row_map[qid], current_user) for qid in created_ids]
    return QuestionBulkResponse(created_count=len(ordered), questions=ordered)


@router.put("/{question_id}", response_model=QuestionResponse)
async def update_question(
    question_id: UUID,
    payload: QuestionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> QuestionResponse:
    question = await db.scalar(
        select(Question)
        .options(selectinload(Question.options), selectinload(Question.category))
        .where(Question.id == question_id)
    )
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found.",
        )
    if question.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to modify this question.",
        )

    if payload.category_id is not None or payload.category_slug is not None:
        category = await _resolve_category(db, payload.category_id, payload.category_slug)
        question.category_id = category.id
    if payload.content is not None:
        question.content = payload.content.strip()
    if payload.code_snippet is not None:
        question.code_snippet = payload.code_snippet
    if payload.code_language is not None:
        question.code_language = payload.code_language
    if payload.explanation is not None:
        question.explanation = payload.explanation.strip()
    if payload.difficulty is not None:
        question.difficulty = payload.difficulty
    if payload.is_public is not None:
        question.is_public = payload.is_public

    if payload.options is not None:
        question.options.clear()
        await db.flush()
        for idx, opt in enumerate(payload.options):
            question.options.append(
                QuestionOption(
                    content=opt.content,
                    is_correct=opt.is_correct,
                    order_index=opt.order_index if opt.order_index is not None else idx,
                )
            )

    await db.commit()
    refreshed = await db.scalar(
        select(Question)
        .options(selectinload(Question.options), selectinload(Question.category))
        .where(Question.id == question_id)
    )
    assert refreshed is not None
    return _to_question_response(refreshed, current_user)


@router.delete("/{question_id}")
async def delete_question(
    question_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    question = await db.get(Question, question_id)
    if question is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Question not found.",
        )
    if question.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this question.",
        )
    await db.delete(question)
    await db.commit()
    return {"detail": "Question deleted."}
