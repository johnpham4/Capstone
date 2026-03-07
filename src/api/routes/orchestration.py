import json
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.rate_limiter import rate_limit_orchestration
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
            user_input=request.user_input,
            mode=request.mode,
            llm_mock=request.llm_mock,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)

        # Persist diagram if present
        if result.get("diagram"):
            d = result["diagram"]
            await history.save_diagram(
                request_id=record.id,
                dsl=d.get("dsl", ""),
                image_base64=d.get("image_base64"),
                generation_time_ms=d.get("generation_time_ms"),
                render_time_ms=d.get("render_time_ms"),
            )

        # Persist solution if present
        if result.get("solution"):
            s = result["solution"]
            await history.save_solution(
                request_id=record.id,
                content=s.get("content", ""),
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



KEEPALIVE_INTERVAL = 15  # seconds between ping events


@router.get("/api/v1/orchestration/stream")
async def stream_orchestration(
    user_input: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    _rate: None = Depends(rate_limit_orchestration),
    mode: str = Query(default="diagram"),
    llm_mock: bool = Query(default=False),
):
    """SSE streaming endpoint — yields events as each pipeline stage completes.

    Event types:
        rewrite  — rewritten problem + resolved mode
        diagram  — generating_dsl → completed (with dsl + image_base64)
        solver   — streaming token chunks → completed
        done     — final summary with request_id
        error    — on any failure
        ping     — keepalive every 15s
    """
    history = HistoryService(db)
    record = await history.create_request(
        user_id=current_user.id,
        input_text=user_input,
        mode=mode,
    )

    async def event_generator():
        start = time.perf_counter()
        collected_dsl = None
        collected_image = None
        collected_solution = None

        try:
            async for event in orchestrator.stream_execute(
                user_input=user_input,
                mode=mode,
                llm_mock=llm_mock,
            ):
                event_type = event.get("event", "unknown")

                # Collect results for persistence
                if event_type == "diagram" and event.get("status") == "completed":
                    collected_dsl = event.get("dsl")
                    collected_image = event.get("image_base64")
                elif event_type == "solver" and event.get("status") == "completed":
                    collected_solution = event.get("solution")

                # SSE format: named event + JSON data
                yield f"event: {event_type}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"

            latency_ms = int((time.perf_counter() - start) * 1000)

            if collected_dsl:
                await history.save_diagram(
                    request_id=record.id,
                    dsl=collected_dsl,
                    image_base64=collected_image,
                    generation_time_ms=latency_ms,
                )
            if collected_solution:
                await history.save_solution(
                    request_id=record.id,
                    content=collected_solution,
                )

            await history.complete_request(record.id, latency_ms=latency_ms)

            # Final event with request_id for frontend to fetch history
            yield f"event: done\ndata: {json.dumps({'request_id': record.id, 'latency_ms': latency_ms})}\n\n"

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            await history.fail_request(record.id, latency_ms=latency_ms)
            logger.exception(f"SSE orchestration stream failed: {e}")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
