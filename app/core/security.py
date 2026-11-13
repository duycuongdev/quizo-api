from datetime import UTC, datetime, timedelta
import hashlib
import re
from typing import Any
from uuid import uuid4

import jwt
from jwt import InvalidTokenError
from pwdlib import PasswordHash

from app.core.config import settings

password_hasher = PasswordHash.recommended()


class TokenVerificationError(Exception):
    """Raised when a JWT token is invalid, expired, or wrong token type."""


def hash_password(password: str) -> str:
    return password_hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    try:
        return password_hasher.verify(plain_password, hashed_password)
    except Exception:
        return False


def validate_password_strength(password: str) -> None:
    if len(password) < 8 or len(password) > 128:
        raise ValueError("Password must be between 8 and 128 characters long.")
    if not re.search(r"[A-Z]", password):
        raise ValueError("Password must contain at least one uppercase letter.")
    if not re.search(r"[a-z]", password):
        raise ValueError("Password must contain at least one lowercase letter.")
    if not re.search(r"\d", password):
        raise ValueError("Password must contain at least one digit.")


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_access_token(subject: str, extra_claims: dict[str, Any] | None = None) -> str:
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": "access",
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    if extra_claims:
        for k, v in extra_claims.items():
            if k not in {"sub", "type", "jti", "iat", "exp"}:
                payload[k] = v
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_refresh_token(subject: str, jti: str | None = None) -> tuple[str, str, datetime]:
    now = datetime.now(UTC)
    expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    token_jti = jti or str(uuid4())
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": "refresh",
        "jti": token_jti,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    encoded = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded, token_jti, expire


def create_email_verification_token(subject: str, email: str) -> str:
    now = datetime.now(UTC)
    expire = now + timedelta(hours=settings.EMAIL_VERIFICATION_EXPIRE_HOURS)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "email": email,
        "type": "email_verification",
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_password_reset_token(subject: str) -> str:
    now = datetime.now(UTC)
    expire = now + timedelta(minutes=settings.PASSWORD_RESET_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "type": "password_reset",
        "jti": str(uuid4()),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str, expected_type: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["sub", "exp", "type", "jti"]},
        )
    except InvalidTokenError as exc:
        raise TokenVerificationError("Invalid or expired token.") from exc

    if payload.get("type") != expected_type:
        raise TokenVerificationError("Invalid token type.")
    if not payload.get("sub"):
        raise TokenVerificationError("Token missing subject.")
    return payload
