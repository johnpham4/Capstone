from typing import Sequence

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from src.repositories.base import AbstractRepository
from src.models.orm import DiagramModel, RequestModel


class DiagramRepository(AbstractRepository[DiagramModel]):

    model = DiagramModel

    async def get_by_request_id(self, request_id: str) -> DiagramModel | None:
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
        stmt = (
            select(DiagramModel)
            .options(joinedload(DiagramModel.request))
            .where(DiagramModel.id == diagram_id)
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
