import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.api.dependencies.auth import get_current_user
from src.api.middleware.rate_limiter import rate_limit_orchestration
from src.infrastructures.database.session import get_db
from src.models.dto.user import User
from src.models.dto.orchestration import OrchestrationRequest
from src.services.orchestration import Orchestrator
from src.services.history import HistoryService
from src.prompts import DSL_INFERENCE_INSTRUCTION


router = APIRouter()
orchestrator = Orchestrator(diagram_prompt=DSL_INFERENCE_INSTRUCTION)


@router.post("/api/v1/orchestration")
async def execute_orchestration(
    request: OrchestrationRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    _rate: None = Depends(rate_limit_orchestration),
):
    history = HistoryService(db)
    record = await history.create_request(
        user_id=current_user.id,
        input_text=request.user_input,
        mode=request.mode,
    )
    start = time.perf_counter()

    try:
        result = await orchestrator.execute(
            request.user_input, mode=request.mode, llm_mock=request.llm_mock,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Persist diagram if present
        if result.get("diagram"):
            d = result["diagram"]
            await history.save_diagram(
                request_id=record.id,
                dsl=d.get("dsl", ""),
                image_base64=d.get("image_base64"),
                model_used=d.get("model_used"),
                generation_time_ms=d.get("generation_time_ms"),
                render_time_ms=d.get("render_time_ms"),
                cache_hit=d.get("cache_hit", False),
            )

        # Persist solution if present
        if result.get("solution"):
            s = result["solution"]
            await history.save_solution(
                request_id=record.id,
                content=s.get("content", ""),
                model_used=s.get("model_used"),
                token_count=s.get("token_count"),
                generation_time_ms=s.get("generation_time_ms"),
                cache_hit=s.get("cache_hit", False),
            )

        await history.complete_request(record.id, latency_ms=latency_ms)

        return {
            "status": "success",
            "request_id": record.id,
            "mode": result.get("mode", request.mode),
            "result": result,
        }

    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)
        await history.fail_request(record.id, latency_ms=latency_ms)
        logger.exception(f"Orchestration failed for request {record.id}")
        raise HTTPException(status_code=500, detail=str(e))
