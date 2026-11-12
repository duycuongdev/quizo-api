import pytest
from httpx import AsyncClient

from app.core.database import normalize_async_database_url


@pytest.mark.asyncio
async def test_health_check_and_security_headers(client: AsyncClient) -> None:
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["database"] == "ok"

    # Security headers must be present
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert response.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "default-src 'none'" in response.headers.get("content-security-policy", "")


def test_normalize_neon_postgres_url() -> None:
    url, connect_args = normalize_async_database_url(
        "postgresql+asyncpg://user:pass@ep-cool.neon.tech/quizo?sslmode=require&channel_binding=require"
    )
    assert url == "postgresql+asyncpg://user:pass@ep-cool.neon.tech/quizo"
    assert connect_args == {"ssl": "require"}

    sqlite_url, sqlite_args = normalize_async_database_url("sqlite+aiosqlite:///./quizo.db")
    assert sqlite_url == "sqlite+aiosqlite:///./quizo.db"
    assert sqlite_args == {"check_same_thread": False}
