from datetime import datetime
from typing import Optional, Sequence

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.repositories.base import AbstractRepository
from src.models.orm import RequestModel


class RequestRepository(AbstractRepository[RequestModel]):
    model = RequestModel

    def _apply_filters(
        self,
        stmt,
        user_id: str,
        *,
        q: Optional[str] = None,
        status: Optional[str] = None,
        mode: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ):
        stmt = stmt.where(RequestModel.user_id == user_id)
        if q:
            stmt = stmt.where(
                or_(
                    RequestModel.input_text.ilike(f"%{q}%"),
                    RequestModel.ocr_text.ilike(f"%{q}%"),
                )
            )
        if status:
            stmt = stmt.where(RequestModel.status == status)
        if mode:
            stmt = stmt.where(RequestModel.mode == mode)
        if from_date:
            stmt = stmt.where(RequestModel.created_at >= from_date)
        if to_date:
            stmt = stmt.where(RequestModel.created_at <= to_date)
        return stmt

    async def get_by_user_id(
        self,
        user_id: str,
        *,
        skip: int = 0,
        limit: int = 20,
        q: Optional[str] = None,
        status: Optional[str] = None,
        mode: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> Sequence[RequestModel]:
        """Lấy danh sách requests của 1 user, mới nhất trước (eager-load relations)."""
        stmt = (
            select(RequestModel)
            .options(
                joinedload(RequestModel.diagram),
                joinedload(RequestModel.solution),
            )
        )
        stmt = self._apply_filters(
            stmt, user_id, q=q, status=status, mode=mode,
            from_date=from_date, to_date=to_date,
        )
        stmt = stmt.order_by(RequestModel.created_at.desc()).offset(skip).limit(limit)
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

    async def count_by_user(
        self,
        user_id: str,
        *,
        q: Optional[str] = None,
        status: Optional[str] = None,
        mode: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> int:
        """Đếm tổng requests của 1 user (có filter)."""
        from sqlalchemy import func

        stmt = select(func.count()).select_from(RequestModel)
        stmt = self._apply_filters(
            stmt, user_id, q=q, status=status, mode=mode,
            from_date=from_date, to_date=to_date,
        )
        result = await self._session.execute(stmt)
        return result.scalar_one()
