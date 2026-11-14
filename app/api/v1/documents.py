import re
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.core.rate_limit import limiter
from app.models.document import Document
from app.models.quiz import Category, Question, QuestionOption
from app.models.user import User
from app.schemas.document import (
    DocumentGenerateRequest,
    DocumentGenerateResponse,
    DocumentResponse,
    DocumentTextCreate,
    GeneratedQuestionItem,
)
from app.services.ai_service import generate_questions_from_text
from app.services.document_parser import (
    parse_uploaded_document,
    sanitize_extracted_text,
    sanitize_filename,
)

router = APIRouter(prefix="/documents", tags=["Documents & AI"])


def _to_document_response(doc: Document) -> DocumentResponse:
    preview = doc.content_text[:250] + ("..." if len(doc.content_text) > 250 else "")
    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        original_filename=doc.original_filename,
        file_type=doc.file_type,
        file_size_bytes=doc.file_size_bytes,
        char_count=doc.char_count,
        preview=preview,
        created_at=doc.created_at,
    )


async def _resolve_or_create_category(
    db: AsyncSession,
    category_id: UUID | None = None,
    category_slug: str | None = None,
    category_name: str | None = None,
    default_title: str = "Tài liệu học tập",
) -> Category:
    """Find category by ID/slug/name, or auto-create one gracefully."""
    category: Category | None = None
    if category_id is not None:
        category = await db.get(Category, category_id)
    elif category_slug:
        category = await db.scalar(select(Category).where(Category.slug == category_slug))
    elif category_name and category_name.strip():
        name_clean = category_name.strip()
        category = await db.scalar(select(Category).where(Category.name.ilike(name_clean)))
        if category is None:
            slug = re.sub(r"[^a-z0-9]+", "-", name_clean.lower()).strip("-") or "danh-muc"
            if await db.scalar(select(Category).where(Category.slug == slug)):
                slug = f"{slug}-{uuid4().hex[:6]}"
            category = Category(
                name=name_clean,
                slug=slug,
                description=f"Danh mục {name_clean}",
                color="#4F46E5",
                icon="folder",
                order_index=1,
            )
            db.add(category)
            await db.flush()

    if category is None:
        category = await db.scalar(select(Category).order_by(Category.order_index).limit(1))

    if category is None:
        name_clean = default_title.strip()[:60] or "Tài liệu học tập"
        slug = re.sub(r"[^a-z0-9]+", "-", name_clean.lower()).strip("-") or "tai-lieu"
        if await db.scalar(select(Category).where(Category.slug == slug)):
            slug = f"{slug}-{uuid4().hex[:6]}"
        category = Category(
            name=name_clean,
            slug=slug,
            description="Danh mục tạo tự động từ tài liệu học tập",
            color="#4F46E5",
            icon="book-open",
            order_index=1,
        )
        db.add(category)
        await db.flush()

    return category


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[DocumentResponse]:
    docs = (
        await db.scalars(
            select(Document)
            .where(Document.user_id == current_user.id)
            .order_by(Document.created_at.desc())
        )
    ).all()
    return [_to_document_response(d) for d in docs]


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None, max_length=255),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    safe_name, file_type, size_bytes, sanitized_text = await parse_uploaded_document(file)
    doc_title = (title or "").strip() or safe_name
    doc = Document(
        user_id=current_user.id,
        title=doc_title[:255],
        original_filename=safe_name,
        file_type=file_type,
        file_size_bytes=size_bytes,
        content_text=sanitized_text,
        char_count=len(sanitized_text),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return _to_document_response(doc)


@router.post("/text", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def create_text_document(
    payload: DocumentTextCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    sanitized_text = sanitize_extracted_text(payload.content_text)
    if len(sanitized_text) < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nội dung tài liệu quá ngắn sau khi lọc bảo mật.",
        )
    doc = Document(
        user_id=current_user.id,
        title=payload.title.strip()[:255],
        original_filename=None,
        file_type="text",
        file_size_bytes=len(sanitized_text.encode("utf-8")),
        content_text=sanitized_text,
        char_count=len(sanitized_text),
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return _to_document_response(doc)


@router.post("/quick-generate", response_model=DocumentGenerateResponse)
@limiter.limit(settings.RATE_LIMIT_AI)
async def quick_generate_from_document(
    request: Request,
    file: UploadFile | None = File(default=None),
    pasted_text: str | None = Form(default=None),
    title: str | None = Form(default=None),
    category_id: UUID | None = Form(default=None),
    category_name: str | None = Form(default=None),
    num_questions: int = Form(default=5),
    difficulty: str = Form(default="mixed"),
    language: str = Form(default="vi"),
    save_to_bank: bool = Form(default=True),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentGenerateResponse:
    """One-stop endpoint: Ingests document file or pasted text -> runs RAG -> saves questions -> returns results."""
    # 1. Extract and sanitize text
    if file is not None and file.filename:
        safe_name, file_type, size_bytes, sanitized_text = await parse_uploaded_document(file)
        doc_title = (title or "").strip() or safe_name
    elif pasted_text and pasted_text.strip():
        sanitized_text = sanitize_extracted_text(pasted_text)
        safe_name = None
        file_type = "text"
        size_bytes = len(sanitized_text.encode("utf-8"))
        doc_title = (title or "").strip() or "Tài liệu ghi chú"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Vui lòng tải lên một file tài liệu (.pdf, .docx, .md, .txt) hoặc dán văn bản nội dung.",
        )

    if len(sanitized_text) < 30:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nội dung tài liệu quá ngắn để có thể sinh câu hỏi trắc nghiệm.",
        )

    # 2. Save Document
    doc = Document(
        user_id=current_user.id,
        title=doc_title[:255],
        original_filename=safe_name,
        file_type=file_type,
        file_size_bytes=size_bytes,
        content_text=sanitized_text,
        char_count=len(sanitized_text),
    )
    db.add(doc)
    await db.flush()

    # 3. Resolve category
    category = await _resolve_or_create_category(
        db=db,
        category_id=category_id,
        category_name=category_name,
        default_title=doc_title,
    )

    # 4. Run RAG Question Generation
    generated = await generate_questions_from_text(
        doc_text=sanitized_text,
        num_questions=max(1, min(20, num_questions)),
        difficulty=difficulty,
        language=language,
        category_name=category.name,
    )

    # 5. Persist to question bank if requested
    output_questions: list[GeneratedQuestionItem] = []
    if save_to_bank:
        for item in generated:
            q = Question(
                category_id=category.id,
                user_id=current_user.id,
                document_id=doc.id,
                content=item.content,
                code_snippet=item.code_snippet,
                code_language=item.code_language,
                explanation=item.explanation,
                source_reference=item.source_reference,
                difficulty=item.difficulty,
                source="ai",
                is_public=False,
            )
            for idx, opt in enumerate(item.options):
                q.options.append(
                    QuestionOption(
                        content=opt.content,
                        is_correct=opt.is_correct,
                        order_index=idx,
                    )
                )
            db.add(q)
            await db.flush()
            output_questions.append(
                item.model_copy(
                    update={
                        "id": q.id,
                        "category_id": category.id,
                        "category_slug": category.slug,
                    }
                )
            )
        await db.commit()
    else:
        await db.commit()
        for item in generated:
            output_questions.append(
                item.model_copy(
                    update={
                        "category_id": category.id,
                        "category_slug": category.slug,
                    }
                )
            )

    return DocumentGenerateResponse(
        document_id=doc.id,
        document_title=doc.title,
        category_id=category.id,
        category_slug=category.slug,
        saved=save_to_bank,
        questions=output_questions,
    )


@router.post("/{document_id}/generate", response_model=DocumentGenerateResponse)
@limiter.limit(settings.RATE_LIMIT_AI)
async def generate_quiz_from_document(
    request: Request,
    document_id: UUID,
    payload: DocumentGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> DocumentGenerateResponse:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy tài liệu.",
        )
    if doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Bạn không có quyền truy cập tài liệu này.",
        )

    category = await _resolve_or_create_category(
        db=db,
        category_id=payload.category_id,
        category_slug=payload.category_slug,
        default_title=doc.title,
    )

    generated = await generate_questions_from_text(
        doc_text=doc.content_text,
        num_questions=payload.num_questions,
        difficulty=payload.difficulty,
        language=payload.language,
        category_name=category.name if category else None,
    )

    output_questions: list[GeneratedQuestionItem] = []
    if payload.save_to_bank and category is not None:
        for item in generated:
            q = Question(
                category_id=category.id,
                user_id=current_user.id,
                document_id=doc.id,
                content=item.content,
                code_snippet=item.code_snippet,
                code_language=item.code_language,
                explanation=item.explanation,
                source_reference=item.source_reference,
                difficulty=item.difficulty,
                source="ai",
                is_public=False,
            )
            for idx, opt in enumerate(item.options):
                q.options.append(
                    QuestionOption(
                        content=opt.content,
                        is_correct=opt.is_correct,
                        order_index=idx,
                    )
                )
            db.add(q)
            await db.flush()
            output_questions.append(
                item.model_copy(
                    update={
                        "id": q.id,
                        "category_id": category.id,
                        "category_slug": category.slug,
                    }
                )
            )
        await db.commit()
    else:
        for item in generated:
            output_questions.append(
                item.model_copy(
                    update={
                        "category_id": category.id if category else None,
                        "category_slug": category.slug if category else None,
                    }
                )
            )

    return DocumentGenerateResponse(
        document_id=doc.id,
        document_title=doc.title,
        category_id=category.id if category else None,
        category_slug=category.slug if category else None,
        saved=payload.save_to_bank,
        questions=output_questions,
    )


@router.delete("/{document_id}")
async def delete_document(
    document_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    doc = await db.get(Document, document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )
    if doc.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this document.",
        )
    await db.delete(doc)
    await db.commit()
    return {"detail": "Document deleted."}
