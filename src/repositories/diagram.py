from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from src.repositories.base import AbstractRepository
from src.models.orm import DiagramModel


class DiagramRepository(AbstractRepository[DiagramModel]):

    model = DiagramModel


    async def get_by_request_id(self, request_id: str) -> DiagramModel | None:
        """Tìm diagram theo request_id.

        Mỗi request chỉ có tối đa 1 diagram (1-1 relationship),
        nên dùng scalar_one_or_none().
        """
        stmt = select(DiagramModel).where(
            DiagramModel.request_id == request_id
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_recent_by_user(
        self,
        user_id: str,
        *,
        limit: int = 20,
    ) -> Sequence[DiagramModel]:
        """Lấy diagrams gần đây của 1 user.

        Join qua bảng requests để lọc theo user_id.
        Order by created_at desc (mới nhất trước).

        Đây là ví dụ về query phức tạp hơn — cần JOIN.
        """
        from src.models.orm import RequestModel

        stmt = (
            select(DiagramModel)
            .join(RequestModel, DiagramModel.request_id == RequestModel.id)
            .where(RequestModel.user_id == user_id)
            .order_by(DiagramModel.created_at.desc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def get_with_request(self, diagram_id: str) -> DiagramModel | None:
        """Lấy diagram kèm eager-load request info.

        Dùng joinedload để tránh N+1 query problem.
        Sau khi query, truy cập diagram.request sẽ không trigger thêm query.
        """
        stmt = (
            select(DiagramModel)
            .options(joinedload(DiagramModel.request))
            .where(DiagramModel.id == diagram_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_cache_hit_rate(self) -> float:
        """Tính tỷ lệ cache hit — dùng cho admin dashboard / analytics.

        Returns:
            Float 0.0 → 1.0 (ví dụ: 0.75 = 75% cache hit).
        """
        from sqlalchemy import func

        total = await self.count()
        if total == 0:
            return 0.0

        stmt = select(func.count()).select_from(DiagramModel).where(
            DiagramModel.cache_hit.is_(True)
        )
        result = await self._session.execute(stmt)
        hits = result.scalar_one()
        return hits / total
