from fastapi import APIRouter
from fastapi.responses import StreamingResponse, Response
from loguru import logger
import json

from .prompt import INSTRUCTION_PROMPT
from src.services.diagram.diagram_generation_service import DiagramService


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
    max_tokens: int = 1024,
    temperature: float = 0.7,
    language: str = "vi"
):
    """
    SSE (Server-Sent Events) streaming endpoint for real-time progress updates.

    This endpoint streams progress updates while generating and rendering diagrams:
    - Event 1: DSL generation started (33%)
    - Event 2: DSL generated, rendering started (66%)
    - Event 3: Rendering completed with image (100%)

    Frontend should use EventSource API to receive these updates.

    Note: EventSource only supports GET requests, so we use query parameters.
    """
    async def event_generator():
        try:
            async for event in diagram_service.stream_pipeline_events(
                user_input=user_input,
                prompt_template=INSTRUCTION_PROMPT,
                max_tokens=max_tokens,
                temperature=temperature,
                epochs=500,
                n_tries=1,
                dpi=150,
            ):
                logger.info(f"SSE event progress={event.get('progress')} status={event.get('status')}")
                yield f"data: {json.dumps(event)}\n\n"

        except Exception as e:
            logger.exception(f"SSE stream failed: {e}")
            yield f"data: {json.dumps({'progress': 0, 'status': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",  # CORS for SSE
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )
