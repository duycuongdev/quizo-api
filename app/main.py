from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.database import AsyncSessionLocal, Base, engine
from app.core.middleware import SecurityHeadersMiddleware, unhandled_exception_handler
from app.core.rate_limit import limiter, rate_limit_exceeded_handler


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Seed initial AI Engineer categories/questions idempotently if available
    try:
        from app.db.seed import seed_database

        async with AsyncSessionLocal() as session:
            await seed_database(session)
    except ImportError:
        pass
    yield


def create_app() -> FastAPI:
    docs_url = "/docs" if settings.ENVIRONMENT != "production" else None
    redoc_url = "/redoc" if settings.ENVIRONMENT != "production" else None
    openapi_url = f"{settings.API_V1_PREFIX}/openapi.json" if settings.ENVIRONMENT != "production" else None

    application = FastAPI(
        title=settings.APP_NAME,
        version="0.1.0",
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    application.add_exception_handler(Exception, unhandled_exception_handler)

    application.add_middleware(SlowAPIMiddleware)
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Accept", "Origin"],
    )

    application.include_router(api_router, prefix=settings.API_V1_PREFIX)
    return application


app = create_app()
