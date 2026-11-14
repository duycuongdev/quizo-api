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

VI_STOPWORDS = {
    "là", "gì", "như", "thế", "nào", "trong", "của", "cho", "và", "với",
    "được", "khi", "tại", "sao", "để", "các", "có", "một", "này", "đó",
    "theo", "về", "ra", "vào", "lại", "thực", "hiện", "cơ", "chế", "sau",
    "đây", "tự", "động", "bị", "bởi", "do", "những", "làm", "phải", "cần",
    "hãy", "hỏi", "chỉ", "nhiều", "mỗi", "hay", "hoặc", "nếu", "thì",
    "what", "which", "how", "the", "and", "for", "with", "when", "why",
}


def clean_text_noise(text: str) -> str:
    """Remove CJK/foreign hallucinated characters and normalize whitespaces."""
    # Strip Chinese/CJK characters (e.g. '而非') accidentally emitted by multilingual LLMs
    cleaned = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf]+", "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def clean_option_text(text: str) -> str:
    """Strip leading 'A.', 'B)', '1.', 'a -' prefixes and clean noise."""
    cleaned = clean_text_noise(text)
    cleaned = re.sub(r"^[A-Da-d1-4][\.\:\)\-]\s*", "", cleaned).strip()
    return cleaned


def clean_question_text(text: str) -> str:
    """Clean question prefix noise such as 'Câu 1:', '1.' etc."""
    cleaned = clean_text_noise(text)
    cleaned = re.sub(r"^(?:Câu\s+[0-9]+|[0-9]+)[\.\:\)\-]\s*", "", cleaned).strip()
    return cleaned


def extract_keywords(text: str) -> set[str]:
    """Extract substantive technical tokens from text for deduplication."""
    tokens = re.findall(r"\b[a-zA-Z0-9_\-\$]+\b", text.lower())
    return {t for t in tokens if len(t) > 1 and t not in VI_STOPWORDS}


def is_duplicate_question(
    new_tokens: set[str], existing_token_sets: list[set[str]]
) -> bool:
    """Determine if a new question is semantically redundant with existing questions."""
    if len(new_tokens) < 4:
        return False
    for existing in existing_token_sets:
        if len(existing) < 4:
            continue
        intersection = new_tokens & existing
        min_len = min(len(new_tokens), len(existing))
        union_len = len(new_tokens | existing)
        overlap_coeff = len(intersection) / min_len
        jaccard = len(intersection) / union_len
        # High overlap of key technical concepts
        if overlap_coeff >= 0.85 and jaccard >= 0.60:
            return True
    return False


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
        "- key_points: Liệt kê đầy đủ mọi điểm kiến thức, khái niệm, công nghệ, tham số, cơ chế, đoạn mã, bug stories và câu hỏi phỏng vấn trong phần này (liệt kê toàn bộ các ý độc lập để phục vụ quét cạn câu hỏi).\n"
        "- is_meta: boolean (true nếu là kế hoạch/lộ trình ôn tập, false nếu là kiến thức kỹ thuật)."
    )
    prompt = (
        "Hãy phân tích và trích xuất danh sách các đầu mục kiến thức cần học trong tài liệu dưới đây:\n\n"
        f"<DOCUMENT>\n{doc_text[:40000]}\n</DOCUMENT>"
    )

    models = [
        settings.GEMINI_MODEL,
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ]
    seen = set()
    candidate_models = [m for m in models if not (m in seen or seen.add(m))]

    last_exc = None
    import time
    for model_name in candidate_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
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
            except Exception as exc:
                last_exc = exc
                time.sleep(1.0)
    if last_exc:
        logger.warning("Extract topics fallback error: %s", last_exc)
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
    difficulty: str,
    language: str,
    category_name: str | None,
) -> list[_GeminiQuestion]:
    """Generate exhaustive substantive questions for a single topic until all knowledge is covered."""
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
        "Bạn là một chuyên gia kỹ thuật và sư phạm cao cấp chuyên thiết kế bài kiểm tra trắc nghiệm chuyên sâu cho kỹ sư phần mềm.\n\n"
        "BỘ QUY TẮC CHUẨN XÁC VỀ KỸ THUẬT VÀ SƯ PHẠM (CRITICAL PEDAGOGICAL & TECHNICAL STANDARDS):\n\n"
        "1. TÍNH CHÍNH XÁC KỸ THUẬT & TÍNH NHẤT QUÁN TUYỆT ĐỐI:\n"
        "   - Socket.IO Flow: Sau khi Client A gọi REST API PUT/PATCH cập nhật DB thành công, Client A mới emit event lên Server -> Server gán senderId và broadcast event 'BE_BOARD_UPDATED' tới các client khác trong room qua `socket.to(boardId).emit(...)`. Tuyệt đối không nhầm server tự emit 'FE_...'.\n"
        "   - dnd-kit Collision Detection: Thuật toán `closestCorners` đo khoảng cách giữa các góc (dùng cho Container/Column lớn), còn `pointerWithin` kiểm tra con trỏ chuột nằm trong droppable nào (dùng cho Card nhỏ). Sensor: `TouchSensor` dùng `delay: 250` để phân biệt drag vs scroll; `MouseSensor` dùng `distance: 10` để tránh click bị coi là drag.\n"
        "   - Google OAuth (@react-oauth/google): Đăng nhập client-side nhận ID token/credential, cấu hình tại Google Cloud Console bắt buộc là 'Authorized JavaScript origins' (không phải 'Authorized redirect URIs').\n"
        "   - MongoDB Ordered Arrays & Aggregation: Ghi đè cả mảng (columnOrderIds, cardOrderIds) là cơ chế Last-Write-Wins (LWW) — đây là một trade-off về thiết kế chứ không ngăn được race condition khi 2 người ghi đồng thời. Trong Aggregation, `$lookup` + `$unwind` KHÔNG bảo toàn thứ tự ban đầu của mảng, cần sắp xếp lại bằng `mapOrder` ở ứng dụng hoặc dùng `$indexOfArray`.\n"
        "   - Cookie sameSite: 'none': Bắt buộc đi kèm `secure: true`. Do cookie tự động gửi trong cross-site request nên CSRF là rủi ro có thật, cần CORS allowlist nghiêm ngặt hoặc Custom Request Header (X-Requested-With, Bearer).\n"
        "   - Redux & Token: Nếu Redux Persist lưu access token vào localStorage, toàn bộ rủi ro XSS trộm token lại xuất hiện, làm mất đi ưu điểm bảo vệ XSS của HttpOnly Cookie.\n"
        "   - Refresh Token Rotation: Single-flight chỉ chống trùng trong 1 tab; khi mở 2 tab cùng lúc sẽ bị coi là token reuse và revoke toàn bộ session. Thao tác revoke token nếu nằm trong transaction bị abort thì blacklist jti sẽ bị rollback.\n"
        "   - Socket Security: Handshake auth là chưa đủ, khi join_room bắt buộc phải check quyền truy cập board (`requireBoardAccess`) để chống IDOR.\n"
        "   - Reverse Proxy: Ứng dụng sau proxy của Render cần `app.set('trust proxy', 1)` để req.ip nhận diện đúng client IP khi rate limit.\n"
        "   - Test scripts: `npm test` chỉ chạy unit test; integration / e2e là script riêng (`npm run test:e2e`).\n\n"
        "2. ĐÁP ÁN NHIỄU (DISTRACTORS INTEGRITY) - TUYỆT ĐỐI TRÁNH ĐÁP ÁN NHIỄU CŨNG ĐÚNG:\n"
        "   - Các đáp án sai (is_correct=false) PHẢI THỰC SỰ SAI hoặc KHÔNG PHÙ HỢP với ngữ cảnh câu hỏi.\n"
        "   - TUYỆT ĐỐI KHÔNG lấy một khẳng định vốn ĐÚNG trong thực tế kỹ thuật để làm đáp án sai (ví dụ: 'Unique index giúp tìm kiếm nhanh hơn' là khẳng định đúng vì nó tạo B-tree index, không được coi nó là sai; 'Render tự restart container khi fail health check' là đúng; 'React.memo cần đi kèm useCallback' là đúng; 'client không dùng được cookie' là lý do hợp lệ...). Nếu cần dùng, hãy biến nó thành một khẳng định sai rõ ràng.\n\n"
        "3. SỰ KHỚP NỐI CÂU HỎI - ĐÁP ÁN (QUESTION-ANSWER ALIGNMENT):\n"
        "   - Khi câu hỏi hỏi 'Giải pháp / Cách khắc phục lỗi X là gì?', đáp án đúng BẮT BUỘC PHẢI LÀ GIẢI PHÁP (Solution), tuyệt đối không đưa nguyên nhân (Root cause) vào làm đáp án đúng.\n"
        "   - Không suy diễn thông tin ngoài tài liệu.\n\n"
        "4. ĐỊNH DẠNG VÀ NGÔN NGỮ (CLEAN FORMAT & LOCALIZATION):\n"
        "   - Trong nội dung đáp án (content của option), TUYỆT ĐỐI KHÔNG thêm chữ cái đầu dòng như 'A. ', 'B. ', 'C. ', 'D. ', '1. ', '2. '.\n"
        "   - TUYỆT ĐỐI KHÔNG để lọt từ ngữ tiếng Trung (như '而非') hay ngôn ngữ lạ. 100% tiếng Việt tự nhiên chuẩn mực kết hợp thuật ngữ kỹ thuật tiếng Anh.\n\n"
        "5. QUY TẮC KHÔNG TRÙNG LẶP (ANTI-REDUNDANCY):\n"
        "   - Mỗi điểm kỹ thuật chỉ tạo 1 câu hỏi sâu sắc, độc lập (ví dụ: đúng 1 câu về Axios interceptor refresh token, 1 câu về JTI blacklist, 1 câu về Bug Story Invite). Tuyệt đối không sinh nhiều biến thể trùng lặp của cùng một câu hỏi.\n"
        "   - CẤU TRÚC: Mỗi câu đúng 4 đáp án, duy nhất 1 đáp án có is_correct=true."
    )

    prompt = (
        f"CHỦ ĐỀ CHUYÊN MÔN CẦN QUÉT CẠN KIẾN THỨC: {topic.title}\n"
        f"Phạm vi tóm tắt: {topic.summary}\n"
        f"Các ý kiến thức trọng tâm: {points_str}\n\n"
        f"YÊU CẦU: Hãy duyệt toàn bộ nội dung của chủ đề này trong tài liệu dưới đây và sinh câu hỏi trắc nghiệm chuyên sâu cho TẤT CẢ các điểm kiến thức, khái niệm, cơ chế, đoạn mã, bug stories và câu hỏi phỏng vấn được đề cập.\n"
        f"Hãy sinh liên tục cho đến khi HẾT LƯỢNG KIẾN THỨC của chủ đề này trong tài liệu (Exhaustive Coverage), tuyệt đối không bỏ sót.\n\n"
        f"Ngôn ngữ: {lang_instruction}\n"
        f"Độ khó: {diff_instruction}\n"
        f"{cat_hint}\n\n"
        f"Nội dung tài liệu nguồn:\n"
        f"<DOCUMENT_SOURCE>\n{doc_text[:45000]}\n</DOCUMENT_SOURCE>"
    )

    models = [
        settings.GEMINI_MODEL,
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ]
    seen = set()
    candidate_models = [m for m in models if not (m in seen or seen.add(m))]

    for model_name in candidate_models:
        for attempt in range(3):
            try:
                response = await client.aio.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=_TopicBatchQuizOutput,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.2,
                    ),
                )
                if response.parsed and isinstance(response.parsed, _TopicBatchQuizOutput):
                    return response.parsed.questions
                if response.text:
                    parsed = _TopicBatchQuizOutput.model_validate_json(response.text)
                    return parsed.questions
            except Exception as exc:
                err_msg = str(exc)
                logger.warning(
                    "Model %s failed for topic '%s' (attempt %d): %s",
                    model_name, topic.title, attempt + 1, err_msg[:120]
                )
                if "503" in err_msg or "429" in err_msg or "UNAVAILABLE" in err_msg:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                break
    return []


async def generate_questions_by_topics(
    doc_text: str,
    all_topics: list[TopicItem],
    selected_topic_ids: list[str],
    difficulty: str = "mixed",
    language: str = "vi",
    category_name: str | None = None,
    density: str = "exhaustive",
) -> list[GeneratedQuestionItem]:
    """Generate comprehensive, exhaustive questions across selected topics in parallel until all knowledge is exhausted."""
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI generation service is not configured.",
        )

    # 1. Filter out meta topics (e.g. "Lộ trình ôn tập 5 ngày")
    substantive_topics = [t for t in all_topics if not getattr(t, "is_meta", False)]
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

    if not target_topics:
        target_topics = [
            TopicItem(
                id="1",
                title="Toàn bộ nội dung tài liệu",
                summary="Tổng hợp toàn diện kiến thức",
                key_points=[],
                is_meta=False,
            )
        ]

    client = genai.Client(
        api_key=settings.GEMINI_API_KEY,
        http_options={"api_version": "v1beta"},
    )

    # 3. Generate exhaustively in parallel across all target topics
    tasks = [
        _generate_single_topic_questions(
            client=client,
            doc_text=doc_text,
            topic=t,
            difficulty=difficulty,
            language=language,
            category_name=category_name,
        )
        for t in target_topics
    ]

    results = await asyncio.gather(*tasks)
    raw_questions: list[_GeminiQuestion] = [q for sublist in results for q in sublist]

    # 4. Normalize, validate output and deduplicate
    validated: list[GeneratedQuestionItem] = []
    seen_contents = set()
    existing_token_sets: list[set[str]] = []
    concept_counts: dict[str, int] = {}

    for q in raw_questions:
        c_clean = clean_question_text(q.content)
        if not c_clean or len(c_clean) < 15:
            continue

        # 4.1 Exact prefix key check
        c_key = re.sub(r"\s+", " ", c_clean.lower())[:80]
        if c_key in seen_contents:
            continue

        # 4.2 Semantic keyword overlap check
        q_tokens = extract_keywords(c_clean)
        if is_duplicate_question(q_tokens, existing_token_sets):
            continue

        # 4.3 Concept frequency capping (Max 2 questions for sensitive recurring concepts)
        concept_hit = None
        sensitive_concepts = [
            "interceptor", "jti", "invite", "closestcorners",
            "pointerwithin", "trust proxy", "touchsensor"
        ]
        for sc in sensitive_concepts:
            if sc in c_clean.lower():
                concept_hit = sc
                break
        if concept_hit:
            current_count = concept_counts.get(concept_hit, 0)
            if current_count >= 2:
                continue
            concept_counts[concept_hit] = current_count + 1

        opts = [
            GeneratedOptionItem(
                content=clean_option_text(o.content),
                is_correct=bool(o.is_correct),
            )
            for o in q.options
            if clean_option_text(o.content)
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

        full_explanation = clean_text_noise(q.explanation)
        if q.source_reference and q.source_reference not in full_explanation:
            full_explanation = f"[Nguồn: {q.source_reference}]\n{full_explanation}"

        seen_contents.add(c_key)
        if q_tokens:
            existing_token_sets.append(q_tokens)

        validated.append(
            GeneratedQuestionItem(
                content=c_clean,
                code_snippet=q.code_snippet.strip() if q.code_snippet else None,
                code_language=q.code_language.strip() if q.code_language else None,
                explanation=full_explanation,
                source_reference=q.source_reference,
                difficulty=q.difficulty,
                options=opts,
            )
        )

    if not validated:
        logger.error("AI service produced 0 validated questions for topics: %s", [t.title for t in target_topics])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Hệ thống AI không thể sinh câu hỏi từ mục này. Vui lòng chọn lại mục khác hoặc thử lại sau giây lát.",
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
        "3. ĐÁP ÁN NHIỄU (DISTRACTORS): Các đáp án sai phải thực sự sai về mặt kỹ thuật, tuyệt đối không dùng mệnh đề vốn đúng trong thực tế làm đáp án sai.\n"
        "4. ALIGNMENT: Hỏi về cách giải quyết/khắc phục thì đáp án phải là giải pháp, không được là nguyên nhân.\n"
        "5. ĐỊNH DẠNG: Tuyệt đối không thêm tiền tố 'A. ', 'B. ' vào đáp án. Không để lẫn chữ Hán hay ngoại ngữ lạ.\n"
        "6. NGUỒN DẪN CHỨNG (source_reference): Mỗi câu hỏi phải ghi rõ lấy từ [Đoạn trích X: Mục ...] nào.\n"
        "7. CẤU TRÚC: Mỗi câu có đúng 4 đáp án (options), trong đó DUY NHẤT 1 đáp án có is_correct=true."
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

    models = [
        settings.GEMINI_MODEL,
        "gemini-3.1-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.6-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ]
    seen = set()
    candidate_models = [m for m in models if not (m in seen or seen.add(m))]

    last_exc = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
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
        except Exception as exc:
            last_exc = exc
            continue

    if last_exc:
        raise last_exc
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
        c_clean = clean_question_text(q.content)
        if not c_clean:
            continue
        opts = [
            GeneratedOptionItem(
                content=clean_option_text(o.content),
                is_correct=bool(o.is_correct)
            )
            for o in q.options
            if clean_option_text(o.content)
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

        full_explanation = clean_text_noise(q.explanation)
        if q.source_reference and q.source_reference not in full_explanation:
            full_explanation = f"[Nguồn: {q.source_reference}]\n{full_explanation}"

        validated.append(
            GeneratedQuestionItem(
                content=c_clean,
                code_snippet=q.code_snippet.strip() if q.code_snippet else None,
                code_language=q.code_language.strip() if q.code_language else None,
                explanation=full_explanation,
                source_reference=q.source_reference,
                difficulty=q.difficulty,
                options=opts,
            )
        )

    if not validated:
        logger.error("AI service produced 0 validated questions for topics: %s", [t.title for t in target_topics])
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Hệ thống AI không thể sinh câu hỏi từ mục này. Vui lòng chọn lại mục khác hoặc thử lại sau giây lát.",
        )

    return all_chunks, validated
