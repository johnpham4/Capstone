from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings.base import settings
from src.infrastructures.database.session import get_db
from src.infrastructures.redis.cache import token_blacklist
from src.models.dto.user import User
from src.models.dto.auth import TokenData
from src.services.auth import AuthService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        username: str | None = payload.get("sub")
        jti: str | None = payload.get("jti")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username, jti=jti)
    except Exception as e:
        raise credentials_exception

    # ── Bước 2: Check blacklist (đã logout chưa) ─────────────────────
    if token_data.jti and await token_blacklist.is_revoked(token_data.jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ── Bước 3: Tìm user trong DB ───────────────────────────────────
    service = AuthService(db)
    user = await service.get_user_by_username(token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Check user không bị disabled. Dùng làm dependency cho protected routes."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

