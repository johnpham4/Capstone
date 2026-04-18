from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_active_user, oauth2_scheme_optional
from src.config.settings import settings
from src.infrastructures.database.session import get_db
from src.models.dto.auth import GoogleLoginRequest, LoginRequest, OtpRequest, OtpVerifyRequest, Token
from src.models.dto.user import User, UserCreate
from src.services.auth import AuthRateLimitError, AuthService

router = APIRouter()


def _set_refresh_cookie(response: Response, refresh_token: str) -> None:
    response.set_cookie(
        key=settings.JWT_REFRESH_COOKIE_NAME,
        value=refresh_token,
        httponly=True,
        secure=settings.JWT_COOKIE_SECURE,
        samesite=settings.JWT_COOKIE_SAMESITE,
        max_age=settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        path="/",
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=settings.JWT_REFRESH_COOKIE_NAME,
        path="/",
    )


@router.post("/api/v1/auth/token", response_model=Token)
async def login_form(
    response: Response,
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Login bằng form (Swagger Authorize button dùng endpoint này)."""
    try:
        service = AuthService(db)
        token, refresh_token = await service.login(form_data.username, form_data.password)
        _set_refresh_cookie(response, refresh_token)
        return token
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/api/v1/auth/login", response_model=Token)
async def login_json(
    response: Response,
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Login bằng JSON (frontend dùng endpoint này)."""
    try:
        service = AuthService(db)
        token, refresh_token = await service.login(body.username, body.password)
        _set_refresh_cookie(response, refresh_token)
        return token
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/api/v1/auth/google", response_model=Token)
async def login_google(
    response: Response,
    body: GoogleLoginRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Login bằng Google ID token (Google One Tap / Google Sign-In)."""
    try:
        service = AuthService(db)
        token, refresh_token = await service.login_with_google(body.id_token)
        _set_refresh_cookie(response, refresh_token)
        return token
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/api/v1/auth/otp/request", status_code=status.HTTP_202_ACCEPTED)
async def request_email_otp(
    body: OtpRequest,
    db: AsyncSession = Depends(get_db),
) -> dict[str, str]:
    try:
        service = AuthService(db)
        await service.request_email_otp(body.email)
        return {"message": "If the email can receive messages, an OTP has been sent."}
    except AuthRateLimitError as e:
        headers = {}
        if e.retry_after_seconds is not None:
            headers["Retry-After"] = str(e.retry_after_seconds)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
            headers=headers,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/api/v1/auth/otp/verify", response_model=Token)
async def verify_email_otp(
    response: Response,
    body: OtpVerifyRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    try:
        service = AuthService(db)
        token, refresh_token = await service.verify_email_otp(body.email, body.code)
        _set_refresh_cookie(response, refresh_token)
        return token
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/api/v1/auth/refresh", response_model=Token)
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> Token:
    refresh_token = request.cookies.get(settings.JWT_REFRESH_COOKIE_NAME)
    if not refresh_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
        )

    try:
        service = AuthService(db)
        token, new_refresh_token = await service.refresh(refresh_token)
        _set_refresh_cookie(response, new_refresh_token)
        return token
    except ValueError as e:
        _clear_refresh_cookie(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request: Request,
    response: Response,
    token: Annotated[str | None, Depends(oauth2_scheme_optional)],
    db: AsyncSession = Depends(get_db),
) -> None:
    """Thu hồi token hiện tại. Sau khi gọi endpoint này, token sẽ không dùng được nữa."""
    refresh_token = request.cookies.get(settings.JWT_REFRESH_COOKIE_NAME)
    service = AuthService(db)
    await service.logout(token=token, refresh_token=refresh_token)
    _clear_refresh_cookie(response)


@router.get("/api/v1/auth/me", response_model=User)
async def me(
    current_user: Annotated[User, Depends(get_current_active_user)],
) -> User:
    return current_user


@router.post("/api/v1/auth/register", response_model=User, status_code=status.HTTP_201_CREATED)
async def register(
    data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Đăng ký user mới."""
    try:
        service = AuthService(db)
        return await service.register(data)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
