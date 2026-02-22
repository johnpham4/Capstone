from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings.base import settings
from src.infrastructures.database.session import get_db
from src.models.dto.user import User
from src.models.dto.auth import TokenData
from src.services.auth import AuthService

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/v1/auth/token")


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: AsyncSession = Depends(get_db),
) -> User:
    """Giải mã JWT token → lấy username → tìm user trong DB.

    Flow:
        1. FastAPI tự extract token từ Header "Authorization: Bearer xxx"
           (nhờ oauth2_scheme)
        2. jwt.decode() giải mã + verify chữ ký + check hết hạn
        3. Lấy username từ payload["sub"]
        4. Query DB để lấy User object

    Nếu bất kỳ bước nào fail → 401 Unauthorized.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    # ── Bước 1: Giải mã JWT ─────────────────────────────────────────
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception

    # ── Bước 2: Tìm user trong DB ───────────────────────────────────
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
