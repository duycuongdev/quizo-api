import asyncio
from datetime import UTC, datetime
from uuid import UUID

from fastapi import HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import (
    TokenVerificationError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token,
    verify_password,
)
from app.models.user import RefreshToken, User


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def _issue_token_pair(db: AsyncSession, user: User) -> tuple[str, str]:
    access_token = create_access_token(
        subject=str(user.id),
        extra_claims={"email": user.email, "provider": user.auth_provider},
    )
    refresh_token, jti, expires_at = create_refresh_token(subject=str(user.id))
    db_token = RefreshToken(
        user_id=user.id,
        jti=jti,
        token_hash=hash_token(refresh_token),
        expires_at=expires_at,
    )
    db.add(db_token)
    await db.commit()
    await db.refresh(user)
    return access_token, refresh_token


async def register_user(
    db: AsyncSession, email: str, password: str, full_name: str
) -> tuple[User, str, str]:
    normalized_email = email.strip().lower()
    existing = await db.scalar(select(User).where(User.email == normalized_email))
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered.",
        )

    user = User(
        email=normalized_email,
        full_name=full_name.strip(),
        hashed_password=hash_password(password),
        auth_provider="local",
    )
    db.add(user)
    await db.flush()
    access_token, refresh_token = await _issue_token_pair(db, user)
    return user, access_token, refresh_token


async def authenticate_local_user(
    db: AsyncSession, email: str, password: str
) -> tuple[User, str, str]:
    normalized_email = email.strip().lower()
    user = await db.scalar(select(User).where(User.email == normalized_email))
    if user is None or not user.hashed_password or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    access_token, refresh_token = await _issue_token_pair(db, user)
    return user, access_token, refresh_token


def _verify_google_token_sync(raw_token: str) -> dict:
    return google_id_token.verify_oauth2_token(
        raw_token,
        google_requests.Request(),
        settings.GOOGLE_CLIENT_ID,
    )


async def authenticate_google_user(
    db: AsyncSession, raw_id_token: str
) -> tuple[User, str, str]:
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google authentication is not configured.",
        )

    try:
        idinfo = await asyncio.to_thread(_verify_google_token_sync, raw_id_token)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token.",
        ) from exc

    issuer = idinfo.get("iss")
    if issuer not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token issuer.",
        )

    if not idinfo.get("email_verified"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google email is not verified.",
        )

    google_sub = idinfo.get("sub")
    email = str(idinfo.get("email", "")).strip().lower()
    if not google_sub or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Google token missing required user claims.",
        )

    user = await db.scalar(select(User).where(User.google_sub == google_sub))
    if user is None:
        user = await db.scalar(select(User).where(User.email == email))
        if user is not None:
            user.google_sub = google_sub
            if not user.avatar_url and idinfo.get("picture"):
                user.avatar_url = str(idinfo["picture"])[:500]
        else:
            full_name = str(idinfo.get("name") or email.split("@")[0]).strip()[:120]
            avatar_url = str(idinfo["picture"])[:500] if idinfo.get("picture") else None
            user = User(
                email=email,
                full_name=full_name or "Quizo User",
                hashed_password=None,
                auth_provider="google",
                google_sub=google_sub,
                avatar_url=avatar_url,
            )
            db.add(user)
            await db.flush()

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive.",
        )

    access_token, refresh_token = await _issue_token_pair(db, user)
    return user, access_token, refresh_token


async def rotate_refresh_token(
    db: AsyncSession, raw_refresh_token: str
) -> tuple[User, str, str]:
    try:
        payload = decode_token(raw_refresh_token, expected_type="refresh")
        user_id = UUID(str(payload["sub"]))
        jti = str(payload["jti"])
    except (TokenVerificationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token.",
        ) from exc

    token_row = await db.scalar(
        select(RefreshToken).where(
            RefreshToken.jti == jti,
            RefreshToken.user_id == user_id,
        )
    )
    if token_row is None or token_row.token_hash != hash_token(raw_refresh_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token.",
        )

    now = datetime.now(UTC)
    if token_row.revoked:
        # Refresh token reuse detected -> revoke all active tokens for this user
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked.is_(False))
            .values(revoked=True, revoked_at=now)
        )
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has already been used or revoked.",
        )

    if _ensure_utc(token_row.expires_at) <= now:
        token_row.revoked = True
        token_row.revoked_at = now
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired.",
        )

    user = await db.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is unavailable.",
        )

    new_refresh_token, new_jti, new_expires_at = create_refresh_token(subject=str(user.id))
    token_row.revoked = True
    token_row.revoked_at = now
    token_row.replaced_by_jti = new_jti

    db.add(
        RefreshToken(
            user_id=user.id,
            jti=new_jti,
            token_hash=hash_token(new_refresh_token),
            expires_at=new_expires_at,
        )
    )
    await db.commit()

    new_access_token = create_access_token(
        subject=str(user.id),
        extra_claims={"email": user.email, "provider": user.auth_provider},
    )
    return user, new_access_token, new_refresh_token


async def revoke_refresh_tokens(
    db: AsyncSession, user_id: UUID, raw_refresh_token: str | None = None
) -> None:
    now = datetime.now(UTC)
    if raw_refresh_token:
        token_h = hash_token(raw_refresh_token)
        await db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.token_hash == token_h,
                RefreshToken.revoked.is_(False),
            )
            .values(revoked=True, revoked_at=now)
        )
    else:
        await db.execute(
            update(RefreshToken)
            .where(
                RefreshToken.user_id == user_id,
                RefreshToken.revoked.is_(False),
            )
            .values(revoked=True, revoked_at=now)
        )
    await db.commit()
