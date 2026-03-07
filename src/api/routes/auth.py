from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_active_user, oauth2_scheme
from src.infrastructures.database.session import get_db
from src.models.dto.auth import LoginRequest, Token
from src.models.dto.user import User, UserCreate
from src.services.auth import AuthService

router = APIRouter()


@router.post("/api/v1/auth/token", response_model=Token)
async def login_form(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Login bằng form (Swagger Authorize button dùng endpoint này)."""
    try:
        service = AuthService(db)
        return await service.login(form_data.username, form_data.password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/api/v1/auth/login", response_model=Token)
async def login_json(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Login bằng JSON (frontend dùng endpoint này)."""
    try:
        service = AuthService(db)
        return await service.login(body.username, body.password)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/api/v1/auth/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db),
) -> None:
    """Thu hồi token hiện tại. Sau khi gọi endpoint này, token sẽ không dùng được nữa."""
    service = AuthService(db)
    await service.logout(token)


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
