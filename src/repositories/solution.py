"""Solution Repository — Data access cho bảng solutions."""

from sqlalchemy import select

from src.repositories.base import AbstractRepository
from src.models.orm import SolutionModel


class SolutionRepository(AbstractRepository[SolutionModel]):
    model = SolutionModel

    async def get_by_request_id(self, request_id: str) -> SolutionModel | None:
        """Tìm solution theo request_id (1-1 relationship)."""
        stmt = select(SolutionModel).where(
            SolutionModel.request_id == request_id
        )
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()
