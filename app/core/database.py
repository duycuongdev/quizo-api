from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy ORM models."""


def normalize_async_database_url(database_url: str) -> tuple[str, dict[str, Any]]:
    """Normalize SQLite and PostgreSQL (including Neon sslmode) URLs for async engines."""
    if database_url.startswith("sqlite"):
        return database_url, {"check_same_thread": False}

    url = database_url
    if url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]

    connect_args: dict[str, Any] = {}
    parsed = urlparse(url)
    if parsed.query:
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        sslmode_vals = query_params.pop("sslmode", [])
        query_params.pop("channel_binding", None)
        if sslmode_vals and sslmode_vals[0] in {"require", "verify-ca", "verify-full"}:
            connect_args["ssl"] = "require"
        new_query = urlencode(
            [(k, v) for k, vals in query_params.items() for v in vals]
        )
        url = urlunparse(parsed._replace(query=new_query))
    return url, connect_args


def _build_engine_kwargs(database_url: str) -> tuple[str, dict[str, Any]]:
    url, connect_args = normalize_async_database_url(database_url)
    kwargs: dict[str, Any] = {
        "echo": settings.DEBUG,
        "future": True,
    }
    if connect_args:
        kwargs["connect_args"] = connect_args
    if not url.startswith("sqlite"):
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 10
    return url, kwargs


def create_app_engine(database_url: str | None = None) -> AsyncEngine:
    raw_url = database_url or settings.DATABASE_URL
    url, kwargs = _build_engine_kwargs(raw_url)
    eng = create_async_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(eng.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
    return eng


engine = create_app_engine()

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
