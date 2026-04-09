from typing import Optional

import boto3
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories import RequestRepository, DiagramRepository, SolutionRepository
from src.models.orm import RequestModel, DiagramModel, SolutionModel
from src.models.dto.history import HistoryItem, HistoryDetail, PaginatedHistory
from src.config.settings.settings import settings


class HistoryService:
    def __init__(self, db: AsyncSession) -> None:
        self._request_repo = RequestRepository(db)
        self._diagram_repo = DiagramRepository(db)
        self._solution_repo = SolutionRepository(db)
        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION or settings.REGION_NAME or "us-east-1",
        )


    async def create_request(
        self,
        user_id: str,
        input_text: str,
        mode: str = "auto",
    ) -> RequestModel:
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
        image_url: Optional[str] = None,
        generation_time_ms: Optional[int] = None,
        render_time_ms: Optional[int] = None,
    ) -> DiagramModel:
        return await self._diagram_repo.create({
            "request_id": request_id,
            "dsl": dsl,
            "image_url": image_url,
            "generation_time_ms": generation_time_ms,
            "render_time_ms": render_time_ms,
        })

    async def save_solution(
        self,
        request_id: str,
        content: str,
    ) -> SolutionModel:
        return await self._solution_repo.create({
            "request_id": request_id,
            "content": content,
        })

    async def complete_request(
        self,
        request_id: str,
        latency_ms: Optional[int] = None,
    ) -> Optional[RequestModel]:
        data: dict = {"status": "completed"}
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return await self._request_repo.update(request_id, data)

    async def fail_request(
        self,
        request_id: str,
        latency_ms: Optional[int] = None,
    ) -> Optional[RequestModel]:
        data: dict = {"status": "failed"}
        if latency_ms is not None:
            data["latency_ms"] = latency_ms
        return await self._request_repo.update(request_id, data)

    async def update_request_mode(
        self,
        request_id: str,
        mode: str,
    ) -> Optional[RequestModel]:
        return await self._request_repo.update(request_id, {"mode": mode})


    async def list_history(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
    ) -> PaginatedHistory:
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
        req = await self._request_repo.get_with_relations(request_id)
        if req is None:
            return None

        image_url = None
        if req.diagram and req.diagram.image_url:
            image_url = self._resolve_diagram_image_url(req.diagram.image_url)

        return HistoryDetail(
            id=req.id,
            input_text=req.input_text,
            mode=req.mode,
            status=req.status,
            latency_ms=req.latency_ms,
            created_at=req.created_at,
            updated_at=req.updated_at,
            dsl=req.diagram.dsl if req.diagram else None,
            image_url=image_url,
            solution=req.solution.content if req.solution else None,
        )

    def _resolve_diagram_image_url(self, value: str) -> str:
        if not value.startswith("s3://"):
            return value

        path = value.removeprefix("s3://")
        if "/" not in path:
            return value

        bucket, key = path.split("/", 1)
        return self._s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )

    async def delete_request(self, request_id: str, user_id: str) -> bool:
        req = await self._request_repo.get_by_id(request_id)
        if req is None or req.user_id != user_id:
            return False
        return await self._request_repo.delete(request_id)
