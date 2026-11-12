import io
from pathlib import PurePosixPath
import re
import zipfile

import docx
from fastapi import HTTPException, UploadFile, status
import pymupdf

from app.core.config import settings

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
MAX_EXTRACTED_CHARS = 150_000
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # 50 MB zip-bomb guard

_INJECTION_PATTERNS = [
    re.compile(r"<\s*/?\s*(system|assistant|developer|instructions|prompt)\s*>", re.IGNORECASE),
    re.compile(r"<\|[a-zA-Z0-9_]+\|>"),
    re.compile(r"\[/?INST\]", re.IGNORECASE),
    re.compile(
        r"(?im)^.*?\b(ignore|disregard|forget)\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts|rules)\b.*$"
    ),
]


def sanitize_filename(filename: str | None) -> str:
    if not filename:
        return "document"
    # Strip any directory components (both Unix and Windows paths)
    base = PurePosixPath(filename.replace("\\", "/")).name
    base = base.replace("\x00", "")
    cleaned = re.sub(r"[^A-Za-z0-9._ -]", "_", base).strip(" .")
    return cleaned[:200] or "document"


def sanitize_extracted_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    # Remove null bytes and ASCII control characters except \n, \r, \t
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw_text)
    # Remove zero-width & directional override chars
    cleaned = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", cleaned)
    # Neutralize prompt injection directives/delimiters
    for pattern in _INJECTION_PATTERNS:
        cleaned = pattern.sub("[filtered-directive]", cleaned)
    # Collapse excessive blank lines
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    cleaned = cleaned.strip()
    if len(cleaned) > MAX_EXTRACTED_CHARS:
        cleaned = cleaned[:MAX_EXTRACTED_CHARS]
    return cleaned


def _extract_pdf(raw_bytes: bytes) -> str:
    if not raw_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid PDF file signature.",
        )
    try:
        with pymupdf.open(stream=raw_bytes, filetype="pdf") as doc:
            pages_text: list[str] = []
            total_len = 0
            for page in doc:
                text = page.get_text("text") or ""
                pages_text.append(text)
                total_len += len(text)
                if total_len >= MAX_EXTRACTED_CHARS:
                    break
            return "\n\n".join(pages_text)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse PDF document.",
        ) from exc


def _extract_docx(raw_bytes: bytes) -> str:
    if not (raw_bytes.startswith(b"PK\x03\x04") or raw_bytes.startswith(b"PK\x05\x06")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid DOCX file signature.",
        )
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            names = set(zf.namelist())
            if "[Content_Types].xml" not in names or not any(
                n.startswith("word/") for n in names
            ):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded ZIP archive is not a valid DOCX file.",
                )
            total_uncompressed = sum(info.file_size for info in zf.infolist())
            if total_uncompressed > MAX_DOCX_UNCOMPRESSED_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="DOCX archive exceeds safe uncompressed size limit.",
                )
    except HTTPException:
        raise
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Corrupted DOCX archive.",
        ) from exc

    try:
        document = docx.Document(io.BytesIO(raw_bytes))
        parts: list[str] = []
        for p in document.paragraphs:
            if p.text.strip():
                parts.append(p.text.strip())
        for table in document.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    parts.append(row_text)
        return "\n".join(parts)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse DOCX document.",
        ) from exc


def _extract_plain_text(raw_bytes: bytes) -> str:
    if b"\x00" in raw_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Binary content is not allowed in text/markdown uploads.",
        )
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text file must be valid UTF-8 encoded.",
        ) from exc


async def parse_uploaded_document(upload: UploadFile) -> tuple[str, str, int, str]:
    """Validate and parse an uploaded document safely in memory.

    Returns: (safe_filename, file_type, size_bytes, sanitized_text)
    """
    safe_name = sanitize_filename(upload.filename)
    ext = PurePosixPath(safe_name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Allowed: .pdf, .docx, .txt, .md",
        )

    max_bytes = settings.max_upload_size_bytes
    chunks: list[bytes] = []
    total_size = 0
    while True:
        chunk = await upload.read(64 * 1024)
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_MB}MB.",
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)
    if total_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if ext == ".pdf":
        extracted = _extract_pdf(raw_bytes)
    elif ext == ".docx":
        extracted = _extract_docx(raw_bytes)
    else:
        extracted = _extract_plain_text(raw_bytes)

    sanitized = sanitize_extracted_text(extracted)
    if len(sanitized) < 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not extract sufficient readable text from document.",
        )

    return safe_name, ext.lstrip("."), total_size, sanitized
