import asyncio
import json
from datetime import datetime, timezone
from typing import Any
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.rate_limiter import rate_limit_orchestration
from src.infrastructures.database.session import get_db
from src.models.dto.user import User
from src.models.dto.orchestration import OrchestrationRequest, OrchestrationResponse
from src.services.orchestration import OrchestrationService, OrchestrationError
from src.services.history import HistoryService
from src.prompts import DSL_INFERENCE_INSTRUCTION


router = APIRouter()

_orchestration_service = OrchestrationService(diagram_prompt=DSL_INFERENCE_INSTRUCTION)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.post("/api/v1/orchestration", response_model=OrchestrationResponse)
async def execute_orchestration(
    request: OrchestrationRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    _rate: None = Depends(rate_limit_orchestration),
):
    try:
        history = HistoryService(db)
        return await _orchestration_service.execute(
            user_id=current_user.id,
            user_input=request.user_input,
            mode=request.mode,
            history=history,
            llm_mock=request.llm_mock,
            image_base64=request.image_base64,
        )

    except OrchestrationError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": exc.code,
                "message": exc.message,
                "request_id": exc.request_id,
            },
        )


@router.post("/api/v1/orchestration/stream")
async def execute_orchestration_stream(
    request: OrchestrationRequest,
    raw_request: Request,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    _rate: None = Depends(rate_limit_orchestration),
):
    history = HistoryService(db)
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def progress_callback(event: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def run_orchestration() -> None:
        try:
            result = await _orchestration_service.execute(
                user_id=current_user.id,
                user_input=request.user_input,
                mode=request.mode,
                history=history,
                llm_mock=request.llm_mock,
                image_base64=request.image_base64,
                progress_callback=progress_callback,
            )
            await queue.put(
                {
                    "event": "orchestration.result",
                    "timestamp": _utcnow_iso(),
                    "request_id": result.get("request_id"),
                    "payload": result,
                }
            )
        except OrchestrationError as exc:
            await queue.put(
                {
                    "event": "orchestration.error",
                    "timestamp": _utcnow_iso(),
                    "request_id": exc.request_id,
                    "error_code": exc.code,
                    "message": exc.message,
                }
            )
        except Exception as exc:
            await queue.put(
                {
                    "event": "orchestration.error",
                    "timestamp": _utcnow_iso(),
                    "error_code": "ORCHESTRATION_EXECUTION_ERROR",
                    "message": str(exc),
                }
            )
        finally:
            await queue.put({"event": "stream.end", "timestamp": _utcnow_iso()})

    orchestration_task = asyncio.create_task(run_orchestration())

    async def event_generator():
        try:
            while True:
                if await raw_request.is_disconnected():
                    if not orchestration_task.done():
                        orchestration_task.cancel()
                    break

                try:
                    event_payload = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                event_name = str(event_payload.get("event", "message"))
                encoded = json.dumps(event_payload, ensure_ascii=True)
                yield f"event: {event_name}\ndata: {encoded}\n\n"

                if event_name == "stream.end":
                    break
        finally:
            if not orchestration_task.done():
                orchestration_task.cancel()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)

