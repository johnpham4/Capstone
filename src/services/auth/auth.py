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
        """Xác thực user, trả JWT token.

        Flow:
            1. Tìm user trong DB theo username
            2. So sánh password (argon2 hash)
            3. Nếu đúng → tạo JWT token chứa username + thời hạn
            4. Trả Token object

        Raises:
            ValueError: Sai username hoặc password.
        """
        user = await self.authenticate(username, password)
        if not user:
            raise ValueError("Incorrect username or password")

        access_token = create_access_token(
            data={"sub": user.username},
            expires_delta=timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        return Token(access_token=access_token, token_type="bearer")

    async def register(self, data: UserCreate) -> User:
        """Tạo user mới.

        Flow:
            1. Check username đã tồn tại chưa
            2. Hash password (argon2)
            3. Lưu vào DB
            4. Trả User schema (không có password)

        Raises:
            ValueError: Username đã tồn tại.
        """
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
        """Thu hồi JWT token bằng cách đưa jti vào Redis blacklist.

        TTL của entry bằng đúng thời gian còn lại của token để tránh
        Redis giữ rác sau khi token đã tự hết hạn.
        """
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
        """Xác thực username + password, trả UserInDB hoặc None.

        Dùng bởi: login(), và có thể dùng bởi bất kỳ flow nào cần verify credentials.
        """
        user = await self.get_user_by_username(username)
        if user is None:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    async def get_user_by_username(self, username: str) -> UserInDB | None:
        """Tìm user theo username, trả Pydantic schema."""
        user_model = await self._repo.get_by_username(username)
        if user_model is None:
            return None
        return self._repo.to_schema(user_model)
