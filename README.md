# Quizo API

Production-grade, security-hardened FastAPI backend for **Quizo** — a PWA AI Engineer quiz platform with AI document-to-quiz generation, manual & bulk question authoring, and SM-2 spaced repetition.

## Tech Stack
- **Framework**: FastAPI + Uvicorn (Python 3.14)
- **Database**: SQLAlchemy 2.0 (async) + Alembic (SQLite `aiosqlite` for dev, PostgreSQL `asyncpg` / Neon ready)
- **Auth & Security**: Argon2 (`pwdlib`) password hashing, short-lived JWT access tokens + rotating/revoked refresh tokens (`PyJWT`), server-side Google OAuth2 `id_token` verification (`google-auth`), `slowapi` rate limiting, strict CORS & security headers middleware
- **AI & Document Parsing**: Google GenAI SDK (`google-genai`), `PyMuPDF` (PDF), `python-docx` (DOCX), TXT/MD with magic-bytes validation & prompt-injection sanitization

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# Edit .env with your SECRET_KEY, GEMINI_API_KEY, GOOGLE_CLIENT_ID
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

## API Overview (`/api/v1`)
- `GET /api/v1/health`
- `POST /api/v1/auth/register`, `login`, `google`, `refresh`, `logout`, `GET /me`
- `GET /api/v1/categories`, `GET /api/v1/categories/{slug}`
- `GET/POST/PUT/DELETE /api/v1/questions`, `POST /api/v1/questions/bulk`
- `GET/POST /api/v1/quizzes`, `GET /api/v1/quizzes/{id}`, `POST /api/v1/quizzes/{id}/submit`, `GET /api/v1/quizzes/attempts/history`, `GET /api/v1/quizzes/attempts/{attempt_id}`
- `GET /api/v1/documents`, `POST /api/v1/documents/upload`, `POST /api/v1/documents/text`, `POST /api/v1/documents/{id}/generate`, `DELETE /api/v1/documents/{id}`
- `GET /api/v1/stats/overview`, `GET /api/v1/stats/srs/due`

## Testing

```powershell
pytest -v
```
