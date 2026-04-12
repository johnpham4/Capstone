import base64
import uuid
from datetime import datetime
from typing import Optional

import boto3
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.repositories import RequestRepository, DiagramRepository, SolutionRepository
from src.models.orm import RequestModel, DiagramModel, SolutionModel
from src.models.dto.history import HistoryItem, HistoryDetail, PaginatedHistory
from src.config.settings import settings


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


    def upload_source_image(self, image_base64: str) -> str | None:
        """Upload user's source image to S3, return s3:// URI or None."""
        if not settings.S3_BUCKET_NAME:
            logger.warning("S3_BUCKET_NAME not configured, skipping source image upload")
            return None
        try:
            image_bytes = base64.b64decode(image_base64)
            key = f"source-images/{uuid.uuid4().hex}.png"
            self._s3_client.put_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=key,
                Body=image_bytes,
                ContentType="image/png",
            )
            return f"s3://{settings.S3_BUCKET_NAME}/{key}"
        except Exception:
            logger.exception("Failed to upload source image to S3")
            return None

    async def create_request(
        self,
        user_id: str,
        input_text: str,
        mode: str = "auto",
        source_image_url: str | None = None,
    ) -> RequestModel:
        return await self._request_repo.create({
            "user_id": user_id,
            "input_text": input_text,
            "mode": mode,
            "status": "processing",
            "source_image_url": source_image_url,
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

    async def update_ocr_text(
        self,
        request_id: str,
        ocr_text: str,
    ) -> Optional[RequestModel]:
        return await self._request_repo.update(request_id, {"ocr_text": ocr_text})


    async def list_history(
        self,
        user_id: str,
        page: int = 1,
        page_size: int = 20,
        q: Optional[str] = None,
        status: Optional[str] = None,
        mode: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> PaginatedHistory:
        offset = (page - 1) * page_size
        filter_kw = dict(q=q, status=status, mode=mode, from_date=from_date, to_date=to_date)
        requests = await self._request_repo.get_by_user_id(
            user_id, limit=page_size, skip=offset, **filter_kw,
        )
        total = await self._request_repo.count_by_user(user_id, **filter_kw)

        items = [
            HistoryItem(
                id=req.id,
                input_text=req.input_text,
                mode=req.mode,
                status=req.status,
                latency_ms=req.latency_ms,
                has_diagram=req.diagram is not None,
                has_solution=req.solution is not None,
                has_source_image=req.source_image_url is not None,
                has_ocr=req.ocr_text is not None,
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

    async def get_detail(self, request_id: str, user_id: str) -> Optional[HistoryDetail]:
        req = await self._request_repo.get_with_relations(request_id)
        if req is None or req.user_id != user_id:
            return None

        image_url = None
        if req.diagram and req.diagram.image_url:
            image_url = self._resolve_diagram_image_url(req.diagram.image_url)

        source_image_url = None
        if req.source_image_url:
            source_image_url = self._resolve_diagram_image_url(req.source_image_url)

        return HistoryDetail(
            id=req.id,
            input_text=req.input_text,
            mode=req.mode,
            status=req.status,
            latency_ms=req.latency_ms,
            created_at=req.created_at,
            updated_at=req.updated_at,
            source_image_url=source_image_url,
            ocr_text=req.ocr_text,
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
