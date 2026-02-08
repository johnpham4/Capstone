from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database.base import Base
from src.infrastructure.database.models import UserModel
from src.infrastructure.database.session import engine
from src.models.schemas.user import UserInDB

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def get_user_from_db(db: AsyncSession, username: str) -> UserInDB | None:
    result = await db.execute(select(UserModel).where(UserModel.username == username))
    user_model = result.scalar_one_or_none()
    if user_model:
        return UserInDB(
            username=user_model.username,
            email=user_model.email,
            full_name=user_model.full_name,
            hashed_password=user_model.hashed_password,
            disabled=user_model.disabled,
        )
    return None

async def create_user_in_db(db: AsyncSession, user_data: dict) -> UserModel:
    user_model = UserModel(**user_data)
    db.add(user_model)
    await db.commit()
    await db.refresh(user_model)
    return user_model
