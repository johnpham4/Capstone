import json
import time
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse, Response
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user
from src.api.middleware.rate_limiter import rate_limit_diagram
from src.infrastructures.database.session import get_db
from src.models.dto.user import User
from src.prompts import DSL_INFERENCE_INSTRUCTION
from src.services.diagram.generation import DiagramService
from src.services.history import HistoryService


router = APIRouter()
diagram_service = DiagramService()


@router.options("/api/v1/diagrams/stream-pipeline")
async def stream_pipeline_options():
    """Handle CORS preflight request for SSE endpoint."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Accept, ngrok-skip-browser-warning",
        }
    )


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
    history = HistoryService(db)
    record = await history.create_request(
        user_id=current_user.id,
        input_text=user_input,
        mode="diagram",
    )

    async def event_generator():
        start = time.perf_counter()
        collected_dsl: Optional[str] = None
        collected_image: Optional[str] = None

        try:
            async for event in diagram_service.stream_pipeline_events(
                user_input=user_input,
                prompt_template=DSL_INFERENCE_INSTRUCTION,
                max_tokens=max_tokens,
                temperature=temperature,
                epochs=500,
                n_tries=1,
                dpi=150,
                llm_mock=llm_mock,
            ):
                logger.info(f"SSE event progress={event.get('progress')} status={event.get('status')}")

                # Capture results as they stream
                if event.get("dsl"):
                    collected_dsl = event["dsl"]
                if event.get("image_base64"):
                    collected_image = event["image_base64"]

                yield f"data: {json.dumps(event)}\n\n"

            latency_ms = int((time.perf_counter() - start) * 1000)
            if collected_dsl:
                await history.save_diagram(
                    request_id=record.id,
                    dsl=collected_dsl,
                    image_base64=collected_image,
                    model_used="sagemaker",
                    generation_time_ms=latency_ms,
                )
            await history.complete_request(record.id, latency_ms=latency_ms)

        except Exception as e:
            latency_ms = int((time.perf_counter() - start) * 1000)
            await history.fail_request(record.id, latency_ms=latency_ms)
            logger.exception(f"SSE stream failed: {e}")
            yield f"data: {json.dumps({'progress': 0, 'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )
