"""HistoryService — business logic for request history tracking.

Responsibilities:
    - Create request records (before orchestration starts)
    - Save diagram / solution results (after completion)
    - Paginated listing, detail view, deletion
"""

import math
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories import RequestRepository, DiagramRepository, SolutionRepository
from src.models.orm import RequestModel, DiagramModel, SolutionModel
from src.models.dto.history import HistoryItem, HistoryDetail, PaginatedHistory


class HistoryService:
    """Orchestrates request-history persistence through repositories."""

    def __init__(self, db: AsyncSession) -> None:
        self._request_repo = RequestRepository(db)
        self._diagram_repo = DiagramRepository(db)
        self._solution_repo = SolutionRepository(db)

    # ── Write operations ─────────────────────────────────────────────

    async def create_request(
        self,
        user_id: str,
        input_text: str,
        mode: str = "auto",
    ) -> RequestModel:
        """Create a new request record with status='processing'."""
        return await self._request_repo.create({
            "user_id": user_id,
            "input_text": input_text,
            "mode": mode,
            "status": "processing",
        })

    async def save_diagram(
        self,
        request_id: str,
        dsl: str,
        image_base64: Optional[str] = None,
        model_used: Optional[str] = None,
        generation_time_ms: Optional[int] = None,
        render_time_ms: Optional[int] = None,
        cache_hit: bool = False,
    ) -> DiagramModel:
        """Persist a generated diagram linked to a request."""
        return await self._diagram_repo.create({
            "request_id": request_id,
            "dsl": dsl,
            "image_base64": image_base64,
            "model_used": model_used,
            "generation_time_ms": generation_time_ms,
            "render_time_ms": render_time_ms,
            "cache_hit": cache_hit,
        })

    async def save_solution(
        self,
        request_id: str,
        content: str,
        model_used: Optional[str] = None,
        token_count: Optional[int] = None,
        generation_time_ms: Optional[int] = None,
        cache_hit: bool = False,
    ) -> SolutionModel:
        """Persist a math solution linked to a request."""
        return await self._solution_repo.create({
            "request_id": request_id,
            "content": content,
            "model_used": model_used,
            "token_count": token_count,
            "generation_time_ms": generation_time_ms,
            "cache_hit": cache_hit,
        })

    async def complete_request(
        self,
        request_id: str,
        latency_ms: Optional[int] = None,
    ) -> Optional[RequestModel]:
        """Mark request as completed with latency."""
        data: dict = {"status": "completed"}
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return await self._request_repo.update(request_id, data)

    async def fail_request(
        self,
        request_id: str,
        latency_ms: Optional[int] = None,
    ) -> Optional[RequestModel]:
        """Mark request as failed."""
        data: dict = {"status": "failed"}
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return await self._request_repo.update(request_id, data)

    # ── Read operations ──────────────────────────────────────────────

    async def list_history(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedHistory:
        """Paginated list of user request history."""
        offset = (page - 1) * page_size
        requests = await self._request_repo.get_by_user_id(
            user_id, limit=page_size, skip=offset
        )
        total = await self._request_repo.count_by_user(user_id)

        items = [
            HistoryItem(
                id=req.id,
                input_text=req.input_text,
                mode=req.mode,
                status=req.status,
                latency_ms=req.latency_ms,
                has_diagram=req.diagram is not None,
                has_solution=req.solution is not None,
                created_at=req.created_at,
            )
            for req in requests
        ]

        return PaginatedHistory(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
        )

    async def get_detail(self, request_id: str) -> Optional[HistoryDetail]:
        """Full detail — flat response, không nested object."""
        req = await self._request_repo.get_with_relations(request_id)
        if req is None:
            return None

        return HistoryDetail(
            id=req.id,
            input_text=req.input_text,
            mode=req.mode,
            status=req.status,
            latency_ms=req.latency_ms,
            created_at=req.created_at,
            updated_at=req.updated_at,
            dsl=req.diagram.dsl if req.diagram else None,
            image_base64=req.diagram.image_base64 if req.diagram else None,
            solution=req.solution.content if req.solution else None,
        )

    async def delete_request(self, request_id: str, user_id: str) -> bool:
        """Delete a request (and cascaded diagram/solution) if owned by user."""
        req = await self._request_repo.get_by_id(request_id)
        if req is None or req.user_id != user_id:
            return False
        return await self._request_repo.delete(request_id)
