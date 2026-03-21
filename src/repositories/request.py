from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.repositories.base import AbstractRepository
from src.models.orm import RequestModel


class RequestRepository(AbstractRepository[RequestModel]):
    model = RequestModel

    async def get_by_user_id(
        self,
        user_id: str,
        *,
        skip: int = 0,
        limit: int = 20,
    ) -> Sequence[RequestModel]:
        """Lấy danh sách requests của 1 user, mới nhất trước (eager-load relations)."""
        stmt = (
            select(RequestModel)
            .options(
                joinedload(RequestModel.diagram),
                joinedload(RequestModel.solution),
            )
            .where(RequestModel.user_id == user_id)
            .order_by(RequestModel.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.unique().scalars().all()

    async def get_with_relations(self, request_id: str) -> RequestModel | None:
        stmt = (
            select(RequestModel)
            .options(
                joinedload(RequestModel.diagram),
                joinedload(RequestModel.solution),
            )
            .where(RequestModel.id == request_id)
        )
        result = await self._session.execute(stmt)
        return result.unique().scalar_one_or_none()

    async def update_status(
        self, request_id: str, status: str, latency_ms: int | None = None
    ) -> RequestModel | None:
        """Cập nhật status và latency sau khi xử lý xong."""
        data: dict = {"status": status}
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return await self.update(request_id, data)

    async def count_by_user(self, user_id: str) -> int:
        """Đếm tổng requests của 1 user."""
        from sqlalchemy import func

        stmt = (
            select(func.count())
            .select_from(RequestModel)
            .where(RequestModel.user_id == user_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one()
