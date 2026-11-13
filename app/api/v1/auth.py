from fastapi import APIRouter, Body, Depends, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.v1.deps import get_current_user
from app.core.config import settings
from app.core.database import get_db
from app.core.rate_limit import limiter
from app.models.user import User
from app.schemas.auth import (
    ForgotPasswordRequest,
    GoogleAuthRequest,
    LoginRequest,
    LogoutRequest,
    RefreshRequest,
    RegisterRequest,
    ResendVerificationRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserResponse,
    VerifyEmailRequest,
)
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


def _build_token_response(user: User, access_token: str, refresh_token: str) -> TokenResponse:
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user),
    )


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def register(
    request: Request,
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user, access_token, refresh_token = await auth_service.register_user(
        db=db,
        email=str(payload.email),
        password=payload.password,
        full_name=payload.full_name,
    )
    return _build_token_response(user, access_token, refresh_token)


@router.post("/login", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login(
    request: Request,
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user, access_token, refresh_token = await auth_service.authenticate_local_user(
        db=db,
        email=str(payload.email),
        password=payload.password,
    )
    return _build_token_response(user, access_token, refresh_token)


@router.post("/google", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def google_login(
    request: Request,
    payload: GoogleAuthRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user, access_token, refresh_token = await auth_service.authenticate_google_user(
        db=db,
        raw_id_token=payload.id_token,
    )
    return _build_token_response(user, access_token, refresh_token)


@router.post("/refresh", response_model=TokenResponse)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def refresh_tokens(
    request: Request,
    payload: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    user, access_token, refresh_token = await auth_service.rotate_refresh_token(
        db=db,
        raw_refresh_token=payload.refresh_token,
    )
    return _build_token_response(user, access_token, refresh_token)


@router.post("/logout")
async def logout(
    payload: LogoutRequest | None = Body(default=None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    await auth_service.revoke_refresh_tokens(
        db=db,
        user_id=current_user.id,
        raw_refresh_token=payload.refresh_token if payload else None,
    )
    return {"detail": "Successfully logged out."}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(current_user)


@router.post("/verify-email")
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def verify_email(
    request: Request,
    payload: VerifyEmailRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    user = await auth_service.verify_email(db=db, token=payload.token)
    return {
        "detail": "Email đã được xác nhận thành công.",
        "user": UserResponse.model_validate(user),
    }


@router.post("/resend-verification")
@limiter.limit("5/minute")
async def resend_verification(
    request: Request,
    payload: ResendVerificationRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    await auth_service.resend_verification_email(db=db, email=str(payload.email))
    return {
        "detail": "Nếu email tồn tại và chưa được kích hoạt, hệ thống đã gửi thư xác nhận mới."
    }


@router.post("/forgot-password")
@limiter.limit("5/minute")
async def forgot_password(
    request: Request,
    payload: ForgotPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    await auth_service.request_password_reset(db=db, email=str(payload.email))
    return {
        "detail": "Nếu email tồn tại trong hệ thống, bạn sẽ nhận được hướng dẫn đặt lại mật khẩu trong vài phút."
    }


@router.post("/reset-password")
@limiter.limit("5/minute")
async def reset_password(
    request: Request,
    payload: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    await auth_service.reset_password(
        db=db,
        token=payload.token,
        new_password=payload.new_password,
    )
    return {
        "detail": "Đặt lại mật khẩu thành công. Vui lòng đăng nhập bằng mật khẩu mới."
    }
