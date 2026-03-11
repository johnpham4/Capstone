from datetime import datetime, timedelta, timezone

import jwt
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings.base import settings
from src.core.security import verify_password, get_password_hash, create_access_token
from src.infrastructures.redis.cache import token_blacklist
from src.models.dto.auth import Token
from src.models.dto.user import User, UserInDB, UserCreate
from src.repositories.user import UserRepository


class AuthService:

    def __init__(self, db: AsyncSession) -> None:
        self._repo = UserRepository(db)

    async def login(self, username: str, password: str) -> Token:
        user = await self.authenticate(username, password)
        if not user:
            raise ValueError("Incorrect username or password")

        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        return Token(access_token=access_token, token_type="bearer")

    async def register(self, data: UserCreate) -> User:
        existing = await self._repo.get_by_username(data.username)
        if existing:
            raise ValueError("Username already registered")

        user_model = await self._repo.create({
            "username": data.username,
            "email": data.email,
            "hashed_password": get_password_hash(data.password),
            "disabled": False,
        })

        return User(
            id=user_model.id,
            username=user_model.username,
            email=user_model.email,
            disabled=user_model.disabled,
        )

    async def logout(self, token: str) -> None:
        try:
            payload = jwt.decode(
                token,
                settings.JWT_SECRET_KEY,
                algorithms=[settings.JWT_ALGORITHM],
            )
            jti: str | None = payload.get("jti")
            exp: int | None = payload.get("exp")
            if jti and exp:
                ttl = int(exp - datetime.now(timezone.utc).timestamp())
                await token_blacklist.revoke(jti, ttl)
        except jwt.PyJWTError:
            # Token đã invalid/hết hạn → không cần blacklist
            pass

    async def authenticate(self, username: str, password: str) -> UserInDB | None:
        user = await self.get_user_by_username(username)
        if user is None:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def get_user_by_username(self, username: str) -> UserInDB | None:
        user_model = await self._repo.get_by_username(username)
        if user_model is None:
            return None
        return self._repo.to_schema(user_model)
