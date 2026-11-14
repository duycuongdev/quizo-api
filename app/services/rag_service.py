"""RAG (Retrieval-Augmented Generation) Service for Document Quiz Generation.

Implements:
1. Semantic & Markdown-aware Document Chunking.
2. Vector Embedding computation via Google Gemini Embedding (gemini-embedding-001).
3. Document Outline & Topic extraction (Nhận diện & loại bỏ các mục meta/lộ trình ôn tập).
4. Exhaustive Multi-Topic Parallel Generation (Sinh đầy đủ 3-5 câu chuyên sâu cho từng chủ đề, bao quát toàn bộ dự án).
5. Grounded Prompt Generation with exact source citations and strict anti-meta filters.
"""

import asyncio
import logging
import math
import re
from typing import Literal

from fastapi import HTTPException, status
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.document import GeneratedOptionItem, GeneratedQuestionItem, TopicItem

logger = logging.getLogger("quizo.rag")


class DocumentChunk(BaseModel):
    index: int
    heading: str
    content: str
    char_count: int
    embedding: list[float] | None = None


class _GeminiOption(BaseModel):
    content: str
    is_correct: bool


class _GeminiQuestion(BaseModel):
    content: str
    code_snippet: str | None = None
    code_language: str | None = None
    explanation: str
    source_reference: str = Field(
        description="Đoạn trích hoặc tiêu đề mục trong tài liệu chứa thông tin này"
    )
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    options: list[_GeminiOption] = Field(min_length=4, max_length=4)


class _GeminiRAGQuizOutput(BaseModel):
    questions: list[_GeminiQuestion]


class _DocumentOutlineOutput(BaseModel):
    document_title: str
    topics: list[TopicItem]


class _TopicBatchQuizOutput(BaseModel):
    questions: list[_GeminiQuestion]


def chunk_document_text(
    text: str, target_size: int = 1200, overlap: int = 150
) -> list[DocumentChunk]:
    """Chunk document text into coherent semantic pieces respecting Markdown headers and paragraphs."""
    lines = text.split("\n")
    chunks: list[DocumentChunk] = []
    current_lines: list[str] = []
    current_len = 0
    current_heading = "Nội dung chung"
    chunk_idx = 1

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current_heading = stripped.lstrip("#").strip()[:100]

        line_len = len(line) + 1
        if current_len + line_len > target_size and current_lines:
            content = "\n".join(current_lines).strip()
            if len(content) >= 30:
                chunks.append(
                    DocumentChunk(
                        index=chunk_idx,
                        heading=current_heading,
                        content=content,
                        char_count=len(content),
                    )
                )
                chunk_idx += 1

            overlap_lines: list[str] = []
            overlap_len = 0
            for l in reversed(current_lines):
                if overlap_len + len(l) + 1 < overlap:
                    overlap_lines.insert(0, l)
                    overlap_len += len(l) + 1
                else:
                    break
            current_lines = overlap_lines
            current_len = overlap_len

        current_lines.append(line)
        current_len += line_len

    if current_lines:
        content = "\n".join(current_lines).strip()
        if len(content) >= 20:
            chunks.append(
                DocumentChunk(
                    index=chunk_idx,
                    heading=current_heading,
                    content=content,
                    char_count=len(content),
                )
            )

    if not chunks and text.strip():
        chunks.append(
            DocumentChunk(
                index=1,
                heading=current_heading,
                content=text.strip(),
                char_count=len(text.strip()),
            )
        )
    return chunks


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def select_relevant_chunks(
    chunks: list[DocumentChunk],
    max_chunks: int = 8,
    focus_topic: str | None = None,
    client: genai.Client | None = None,
) -> list[DocumentChunk]:
    """Select the most representative or topic-focused chunks using RAG retrieval."""
    if len(chunks) <= max_chunks:
        return chunks

    if focus_topic and client and settings.GEMINI_API_KEY:
        try:
            topic_embed_res = client.models.embed_content(
                model="gemini-embedding-001", contents=focus_topic
            )
            topic_vec = topic_embed_res.embeddings[0].values

            scored = []
            for ch in chunks:
                if ch.embedding is None:
                    res = client.models.embed_content(
                        model="gemini-embedding-001", contents=ch.content[:1000]
                    )
                    ch.embedding = res.embeddings[0].values
                sim = _cosine_similarity(topic_vec, ch.embedding)
                scored.append((sim, ch))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:max_chunks]]
        except Exception as exc:
            logger.warning("RAG embedding retrieval fallback to stratified sampling: %s", exc)

    step = len(chunks) / max_chunks
    selected: list[DocumentChunk] = []
    for i in range(max_chunks):
        idx = min(int(i * step), len(chunks) - 1)
        if chunks[idx] not in selected:
            selected.append(chunks[idx])
    return selected


def _call_gemini_extract_topics_sync(doc_text: str) -> tuple[str, list[TopicItem]]:
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options={"api_version": "v1beta"},
    )
    system_instruction = (
        "Bạn là một chuyên gia phân tích tài liệu kỹ thuật và thiết kế chương trình học.\n"
        "Nhiệm vụ: Đọc hiểu tài liệu và phân tích thành danh sách các ĐẦU MỤC KIẾN THỨC CỐT LÕI (Topics) để kiểm tra kiến thức.\n\n"
        "QUY TẮC ĐẶC BIỆT VỀ META (is_meta):\n"
        "- Nếu một mục chỉ là lộ trình thời gian (ví dụ 'Lộ trình ôn tập 5 ngày', 'Kế hoạch học theo ngày', 'Lịch trình tuần'), "
        "mục lục, lời mở đầu, ghi chú tác giả hoặc hướng dẫn học tập bên lề (KHÔNG chứa kiến thức kỹ thuật chuyên môn thực chất), "
        "BẮT BUỘC ĐÁNH DẤU is_meta=true để hệ thống bỏ qua không sinh câu hỏi trắc nghiệm.\n"
        "- Các mục còn lại chứa kiến thức chuyên môn, công nghệ, kiến trúc, cơ chế, code, bug stories thì is_meta=false.\n\n"
        "Yêu cầu cho mỗi TopicItem:\n"
        "- id: Chuỗi số thứ tự '1', '2', '3',...\n"
        "- title: Tên tiêu đề ngắn gọn (ví dụ: 'Kiến trúc Backend & MongoDB', 'Authentication & JWT', v.v.).\n"
        "- summary: Tóm tắt 1-2 câu về phạm vi kiến thức trong phần này.\n"
        "- key_points: 2 đến 4 ý kiến thức chuyên môn cốt lõi được đề cập.\n"
        "- is_meta: boolean (true nếu là kế hoạch/lộ trình ôn tập, false nếu là kiến thức kỹ thuật)."
    )
    prompt = (
        "Hãy phân tích và trích xuất danh sách các đầu mục kiến thức cần học trong tài liệu dưới đây:\n\n"
        f"<DOCUMENT>\n{doc_text[:40000]}\n</DOCUMENT>"
    )
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=_DocumentOutlineOutput,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.2,
        ),
    )
    if response.parsed and isinstance(response.parsed, _DocumentOutlineOutput):
        return response.parsed.document_title, response.parsed.topics
    if response.text:
        parsed = _DocumentOutlineOutput.model_validate_json(response.text)
        return parsed.document_title, parsed.topics
    return "Tài liệu học tập", []


async def extract_document_topics(doc_text: str) -> tuple[str, list[TopicItem]]:
    """Analyze document and extract structured learning outline/topics with meta-flagging."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI service is not configured.",
        )
    try:
        return await asyncio.to_thread(_call_gemini_extract_topics_sync, doc_text)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to extract document topics: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Không thể phân tích đầu mục tài liệu: {str(exc)}",
        ) from exc


async def _generate_single_topic_questions(
    client: genai.Client,
    doc_text: str,
    topic: TopicItem,
    q_count: int,
    difficulty: str,
    language: str,
    category_name: str | None,
) -> list[_GeminiQuestion]:
    """Generate high-depth substantive questions for a single topic using Gemini."""
    lang_instruction = (
        "Soạn câu hỏi và đáp án bằng Tiếng Việt tự nhiên, chuẩn mực giáo dục. "
        "Giữ nguyên thuật ngữ kỹ thuật tiếng Anh (REST, JWT, React, Docker, Socket.IO, v.v.)."
        if language == "vi"
        else "Generate questions and explanations completely in clear, standard English."
    )
    diff_instruction = {
        "easy": "Mức độ DỄ: Tập trung vào định nghĩa, vai trò và thông số kỹ thuật.",
        "medium": "Mức độ TRUNG BÌNH: Đòi hỏi hiểu cơ chế hoạt động, so sánh, ưu/nhược điểm.",
        "hard": "Mức độ KHÓ: Phân tích sâu, xử lý race condition/memory leak, kiến trúc dự án thực tế.",
        "mixed": "Mức độ TỔNG HỢP: Kết hợp đa dạng câu hỏi từ cơ chế đến tình huống thực tế.",
    }.get(difficulty, "Mức độ TỔNG HỢP.")

    cat_hint = f"Lĩnh vực: {category_name}." if category_name else ""
    points_str = "; ".join(topic.key_points) if topic.key_points else ""

    system_instruction = (
        "Bạn là một chuyên gia kỹ thuật và sư phạm tạo câu hỏi trắc nghiệm chuyên sâu.\n"
        "QUY TẮC CỐT LÕI BẮT BUỘC:\n"
        "1. CHỈ HỎI VỀ KIẾN THỨC CHUYÊN MÔN THỰC CHẤT: Kiến trúc hệ thống, cơ chế hoạt động, công nghệ, "
        "xử lý ngoại lệ, nguyên nhân & giải pháp sửa lỗi (bug stories), thiết kế database, tối ưu hiệu năng, mã code.\n"
        "2. TUYỆT ĐỐI NGHIÊM CẤM sinh câu hỏi về các nội dung meta như: Lộ trình ôn tập (ví dụ: ngày 1 làm gì, ngày 2 làm gì), "
        "kế hoạch học tập, thời gian biểu, mục lục, tác giả, hoặc hướng dẫn đọc tài liệu.\n"
        "3. BÁM SÁT TÀI LIỆU (GROUNDING): Toàn bộ câu hỏi, đáp án và giải thích phải bám sát 100% vào nội dung tài liệu nguồn.\n"
        "4. DẪN CHỨNG (source_reference): Ghi rõ câu hỏi lấy từ mục/đoạn nào trong tài liệu.\n"
        "5. CẤU TRÚC: Mỗi câu đúng 4 đáp án, duy nhất 1 đáp án có is_correct=true."
    )

    prompt = (
        f"CHỦ ĐỀ CHUYÊN MÔN CẦN RA ĐỀ: {topic.title}\n"
        f"Phạm vi tóm tắt: {topic.summary}\n"
        f"Kiến thức trọng tâm: {points_str}\n\n"
        f"YÊU CẦU: Hãy tạo đúng {q_count} câu hỏi trắc nghiệm chuyên sâu về chủ đề '{topic.title}'.\n"
        f"Ngôn ngữ: {lang_instruction}\n"
        f"Độ khó: {diff_instruction}\n"
        f"{cat_hint}\n\n"
        f"Nội dung tài liệu nguồn:\n"
        f"<DOCUMENT_SOURCE>\n{doc_text[:45000]}\n</DOCUMENT_SOURCE>"
    )

    try:
        response = await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=_TopicBatchQuizOutput,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                temperature=0.3,
            ),
        )
        if response.parsed and isinstance(response.parsed, _TopicBatchQuizOutput):
            return response.parsed.questions
        if response.text:
            parsed = _TopicBatchQuizOutput.model_validate_json(response.text)
            return parsed.questions
    except Exception as exc:
        logger.warning("Error generating questions for topic '%s': %s", topic.title, exc)
    return []


async def generate_questions_by_topics(
    doc_text: str,
    all_topics: list[TopicItem],
    selected_topic_ids: list[str],
    difficulty: str = "mixed",
    language: str = "vi",
    category_name: str | None = None,
    density: str = "comprehensive",
) -> list[GeneratedQuestionItem]:
    """Generate comprehensive, exhaustive questions across selected topics in parallel."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI generation service is not configured.",
        )

    # 1. Filter out meta topics (e.g. "Lộ trình ôn tập 5 ngày")
    substantive_topics = [t for t in all_topics if not getattr(t, "is_meta", False)]
    # Fallback if all topics were flagged as meta
    if not substantive_topics:
        substantive_topics = all_topics

    # 2. Determine target topics
    is_all = (
        not selected_topic_ids
        or "all" in selected_topic_ids
        or len(selected_topic_ids) >= len(all_topics)
    )

    if is_all:
        target_topics = substantive_topics
    else:
        target_topics = [
            t for t in substantive_topics if str(t.id) in selected_topic_ids
        ]
        if not target_topics:
            target_topics = substantive_topics

    # 3. Determine questions per topic dynamically
    # If 1 topic chosen: 6-8 questions
    # If 2-4 topics chosen: 4-5 questions per topic
    # If 5+ topics chosen: 3-4 questions per topic (e.g. 10 topics -> 30-40 questions)
    num_topics = len(target_topics)
    if density == "focused":
        q_per_topic = 2 if num_topics >= 6 else 3
    else:
        # Comprehensive
        if num_topics == 1:
            q_per_topic = 6
        elif num_topics <= 3:
            q_per_topic = 5
        elif num_topics <= 6:
            q_per_topic = 4
        else:
            q_per_topic = 3

    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options={"api_version": "v1beta"},
    )

    # 4. Generate in parallel using asyncio.gather
    tasks = [
        _generate_single_topic_questions(
            client=client,
            doc_text=doc_text,
            topic=t,
            q_count=q_per_topic,
            difficulty=difficulty,
            language=language,
            category_name=category_name,
        )
        for t in target_topics
    ]

    results = await asyncio.gather(*tasks)
    raw_questions: list[_GeminiQuestion] = [q for sublist in results for q in sublist]

    # 5. Normalize and validate output
    validated: list[GeneratedQuestionItem] = []
    for q in raw_questions:
        opts = [
            GeneratedOptionItem(content=o.content.strip(), is_correct=bool(o.is_correct))
            for o in q.options
            if o.content.strip()
        ]
        if len(opts) < 2:
            continue
        correct_indices = [i for i, o in enumerate(opts) if o.is_correct]
        if len(correct_indices) != 1:
            chosen = correct_indices[0] if correct_indices else 0
            opts = [
                GeneratedOptionItem(content=o.content, is_correct=(idx == chosen))
                for idx, o in enumerate(opts)
            ]

        full_explanation = q.explanation.strip()
        if q.source_reference and q.source_reference not in full_explanation:
            full_explanation = f"[Nguồn: {q.source_reference}]\n{full_explanation}"

        validated.append(
            GeneratedQuestionItem(
                content=q.content.strip(),
                code_snippet=q.code_snippet.strip() if q.code_snippet else None,
                code_language=q.code_language.strip() if q.code_language else None,
                explanation=full_explanation,
                source_reference=q.source_reference,
                difficulty=q.difficulty,
                options=opts,
            )
        )

    if not validated:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service returned no valid questions.",
        )

    return validated


def _call_gemini_rag_sync(
    retrieved_chunks: list[DocumentChunk],
    num_questions: int,
    difficulty: str,
    language: str,
    category_name: str | None,
    focus_topic: str | None,
) -> _GeminiRAGQuizOutput:
    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options={"api_version": "v1beta"},
    )

    lang_instruction = (
        "Soạn câu hỏi và đáp án hoàn toàn bằng Tiếng Việt tự nhiên, súc tích và chuẩn mực giáo dục. "
        "Giữ nguyên các thuật ngữ kỹ thuật, tên công nghệ, từ khóa lập trình (như JWT, React, Docker, REST, v.v.)."
        if language == "vi"
        else "Generate questions and explanations completely in clear, standard English."
    )

    diff_instruction = {
        "easy": "Mức độ DỄ: Tập trung vào định nghĩa, khái niệm cơ bản, vai trò và thông số được nhắc trực tiếp trong tài liệu.",
        "medium": "Mức độ TRUNG BÌNH: Đòi hỏi hiểu cơ chế hoạt động, so sánh, ưu/nhược điểm hoặc ứng dụng kiến thức trong tài liệu.",
        "hard": "Mức độ KHÓ: Câu hỏi tình huống, phân tích sâu, xử lý ngoại lệ hoặc giải thích bản chất kiến trúc phức tạp.",
        "mixed": "Mức độ TỔNG HỢP: Kết hợp đa dạng câu dễ, trung bình và khó để kiểm tra toàn diện người học.",
    }.get(difficulty, "Mức độ TRUNG BÌNH.")

    cat_hint = f"Lĩnh vực chuyên môn: {category_name}." if category_name else ""
    topic_hint = f"Chủ đề trọng tâm cần tập trung: {focus_topic}." if focus_topic else ""

    chunks_formatted = []
    for ch in retrieved_chunks:
        chunks_formatted.append(
            f"--- [ĐOẠN TRÍCH {ch.index}: Mục '{ch.heading}'] ---\n{ch.content}"
        )
    context_blob = "\n\n".join(chunks_formatted)

    system_instruction = (
        "Bạn là một chuyên gia giáo dục tạo câu hỏi trắc nghiệm khách quan dựa trên kỹ thuật RAG (Retrieval-Augmented Generation).\n"
        "QUY TẮC RAG CỐT LÕI BẮT BUỘC:\n"
        "1. GROUNDING TUYỆT ĐỐI: Tất cả câu hỏi, đáp án đúng và giải thích PHẢI BÁM SÁT 100% VÀO DỮ LIỆU CÓ TRONG <DOCUMENT_CHUNKS>.\n"
        "2. CHỈ HỎI VỀ KIẾN THỨC CHUYÊN MÔN THỰC CHẤT. TUYỆT ĐỐI KHÔNG sinh câu hỏi về lộ trình ôn tập, thời gian biểu, mục lục.\n"
        "3. NGUỒN DẪN CHỨNG (source_reference): Mỗi câu hỏi phải ghi rõ lấy từ [Đoạn trích X: Mục ...] nào.\n"
        "4. LỜI GIẢI THÍCH (explanation): Phải chỉ rõ vì sao đáp án đúng dựa vào trích dẫn câu chữ cụ thể trong tài liệu.\n"
        "5. CẤU TRÚC: Mỗi câu có đúng 4 đáp án (options), trong đó DUY NHẤT 1 đáp án có is_correct=true."
    )

    user_prompt = (
        f"Hãy tạo đúng {num_questions} câu hỏi trắc nghiệm chất lượng cao từ các đoạn trích tài liệu dưới đây.\n"
        f"Ngôn ngữ: {lang_instruction}\n"
        f"Độ khó: {diff_instruction}\n"
        f"{cat_hint}\n"
        f"{topic_hint}\n\n"
        f"<DOCUMENT_CHUNKS>\n"
        f"{context_blob}\n"
        f"</DOCUMENT_CHUNKS>"
    )

    response = client.models.generate_content(
        model=settings.GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=_GeminiRAGQuizOutput,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            temperature=0.3,
        ),
    )

    if response.parsed is not None and isinstance(response.parsed, _GeminiRAGQuizOutput):
        return response.parsed
    if response.text:
        return _GeminiRAGQuizOutput.model_validate_json(response.text)
    raise ValueError("Empty response from Gemini API.")


async def generate_rag_questions_from_document(
    doc_text: str,
    num_questions: int = 5,
    difficulty: str = "mixed",
    language: str = "vi",
    category_name: str | None = None,
    focus_topic: str | None = None,
) -> tuple[list[DocumentChunk], list[GeneratedQuestionItem]]:
    """Complete RAG pipeline: chunking -> retrieval -> grounded Gemini generation."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI generation service is not configured.",
        )

    all_chunks = chunk_document_text(doc_text)
    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document contains no readable text to chunk.",
        )

    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options={"api_version": "v1beta"},
    )
    retrieved_chunks = select_relevant_chunks(
        all_chunks,
        max_chunks=8,
        focus_topic=focus_topic,
        client=client,
    )

    try:
        raw_output = await asyncio.to_thread(
            _call_gemini_rag_sync,
            retrieved_chunks,
            num_questions,
            difficulty,
            language,
            category_name,
            focus_topic,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Gemini RAG generation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI RAG service failed: {str(exc)}",
        ) from exc

    validated: list[GeneratedQuestionItem] = []
    for q in raw_output.questions[:num_questions]:
        opts = [
            GeneratedOptionItem(content=o.content.strip(), is_correct=bool(o.is_correct))
            for o in q.options
            if o.content.strip()
        ]
        if len(opts) < 2:
            continue
        correct_indices = [i for i, o in enumerate(opts) if o.is_correct]
        if len(correct_indices) != 1:
            chosen = correct_indices[0] if correct_indices else 0
            opts = [
                GeneratedOptionItem(content=o.content, is_correct=(idx == chosen))
                for idx, o in enumerate(opts)
            ]

        full_explanation = q.explanation.strip()
        if q.source_reference and q.source_reference not in full_explanation:
            full_explanation = f"[Nguồn: {q.source_reference}]\n{full_explanation}"

        validated.append(
            GeneratedQuestionItem(
                content=q.content.strip(),
                code_snippet=q.code_snippet.strip() if q.code_snippet else None,
                code_language=q.code_language.strip() if q.code_language else None,
                explanation=full_explanation,
                source_reference=q.source_reference,
                difficulty=q.difficulty,
                options=opts,
            )
        )

    if not validated:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI service returned no valid questions.",
        )

    return all_chunks, validated
