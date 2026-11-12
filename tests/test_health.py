import pytest
from httpx import ASGITransport, AsyncClient

from app.core.database import normalize_async_database_url
from app.main import create_app


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


@pytest.mark.asyncio
async def test_cors_whitelist_and_no_stack_trace_leak() -> None:
    test_app = create_app()

    @test_app.get("/api/v1/_boom_test")
    async def _boom() -> None:
        raise RuntimeError("sensitive internal connection string leaked")

    async with AsyncClient(
        transport=ASGITransport(app=test_app, raise_app_exceptions=False),
        base_url="http://testserver",
    ) as ac:
        # Allowed origin
        ok_cors = await ac.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert ok_cors.headers.get("access-control-allow-origin") == "http://localhost:3000"

        # Disallowed origin
        bad_cors = await ac.options(
            "/api/v1/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert bad_cors.headers.get("access-control-allow-origin") is None

        # Unhandled error never leaks raw stack trace or message
        err_res = await ac.get("/api/v1/_boom_test")
        assert err_res.status_code == 500
        assert err_res.json() == {"detail": "Internal server error"}
        assert "sensitive internal" not in err_res.text
        assert "Traceback" not in err_res.text


def test_normalize_neon_postgres_url() -> None:
    url, connect_args = normalize_async_database_url(
        "postgresql+asyncpg://user:pass@ep-cool.neon.tech/quizo?sslmode=require&channel_binding=require"
    )
    assert url == "postgresql+asyncpg://user:pass@ep-cool.neon.tech/quizo"
    assert connect_args == {"ssl": "require"}

    sqlite_url, sqlite_args = normalize_async_database_url("sqlite+aiosqlite:///./quizo.db")
    assert sqlite_url == "sqlite+aiosqlite:///./quizo.db"
    assert sqlite_args == {"check_same_thread": False}
