"""AI Service wrapping the RAG Question Generation pipeline with Gemini."""

from app.schemas.document import GeneratedQuestionItem
from app.services.rag_service import generate_rag_questions_from_document


async def generate_questions_from_text(
    doc_text: str,
    num_questions: int,
    difficulty: str,
    language: str,
    category_name: str | None = None,
    focus_topic: str | None = None,
) -> list[GeneratedQuestionItem]:
    """Generate high-quality grounded quiz questions from document text via RAG."""
    _chunks, questions = await generate_rag_questions_from_document(
        doc_text=doc_text,
        num_questions=num_questions,
        difficulty=difficulty,
        language=language,
        category_name=category_name,
        focus_topic=focus_topic,
    )
    return questions
