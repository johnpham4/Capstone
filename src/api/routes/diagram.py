import time
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi import HTTPException
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.rate_limiter import rate_limit_diagram
from src.infrastructures.database.session import get_db
from src.models.dto.user import User
from src.services.diagram.generation import DiagramService
from src.services.history import HistoryService


router = APIRouter()
diagram_service = DiagramService()


@router.get("/api/v1/diagrams/stream-pipeline")
async def stream_pipeline(
    user_input: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    _rate: None = Depends(rate_limit_diagram),
    max_tokens: int = Query(default=1024, ge=100, le=4096),
    temperature: float = Query(default=0.7, ge=0.0, le=2.0),
    language: str = Query(default="vi"),
    llm_mock: bool = Query(default=False),
):
    _ = (max_tokens, temperature, language, llm_mock)
    history = HistoryService(db)
    record = await history.create_request(
        user_id=current_user.id,
        input_text=user_input,
        mode="diagram",
    )
    start = time.perf_counter()

    try:
        result = diagram_service.generate_and_render(
            task_id=f"diagram_{record.id}",
            dsl=user_input,
            epochs=500,
            n_tries=1,
            dpi=150,
        )

        if result.get("status") == "failed":
            raise RuntimeError(result.get("error", "Diagram generation failed"))

        latency_ms = int((time.perf_counter() - start) * 1000)
        await history.save_diagram(
            request_id=record.id,
            dsl=result.get("dsl", user_input),
            image_base64=result.get("image_base64"),
            generation_time_ms=latency_ms,
        )
        await history.complete_request(record.id, latency_ms=latency_ms)

        return {
            "status": "success",
            "request_id": record.id,
            "result": result,
        }
    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        await history.fail_request(record.id, latency_ms=latency_ms)
        logger.exception(f"Diagram generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
