from loguru import logger
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories.base import AbstractRepository
from src.models.orm import UserModel
from src.models.dto.user import UserInDB


class UserRepository(AbstractRepository[UserModel]):

    model = UserModel

    async def get_by_username(self, username: str) -> UserModel | None:
        stmt = select(UserModel).where(UserModel.username == username)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> UserModel | None:
        normalized = email.strip().lower()
        stmt = (
            select(UserModel)
            .where(func.lower(UserModel.email) == normalized)
            .order_by(UserModel.created_at.asc(), UserModel.id.asc())
            .limit(2)
        )
        result = await self._session.execute(stmt)
        users = result.scalars().all()

        if len(users) > 1:
            logger.warning(f"Duplicate users found for email={normalized}. Using the oldest account.")

        return users[0] if users else None

    @staticmethod
    def to_schema(user: UserModel) -> UserInDB:
        return UserInDB(
            id=user.id,
            username=user.username,
            email=user.email,
            hashed_password=user.hashed_password,
            disabled=user.disabled,
        )
