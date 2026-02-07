from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse, Response
from uuid import uuid4
from loguru import logger
import boto3
import json
import asyncio

from .prompt import INSTRUCTION_PROMPT
from src.config.settings.base import settings


router = APIRouter()

sagemaker_client = boto3.client(
    'sagemaker-runtime',
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
)


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
            request_id = str(uuid4())

            # Event 1: Starting DSL generation
            logger.info(f"[{request_id}] SSE: Starting DSL generation")
            yield f"data: {json.dumps({'progress': 10, 'status': 'Generating diagram code...', 'request_id': request_id})}\n\n"

            # Step 1: Generate DSL via SageMaker
            full_prompt = INSTRUCTION_PROMPT.format(query=user_input)
            payload = {
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "top_k": 50
            }

            response = sagemaker_client.invoke_endpoint(
                EndpointName=settings.SAGEMAKER_ENDPOINT_INFERENCE,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            result = json.loads(response['Body'].read().decode('utf-8'))

            dsl_output = ""
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                dsl_output = choice.get('message', {}).get('content', '') or choice.get('text', '')
            else:
                dsl_output = result.get("generated_text", result.get("text", ""))

            if not dsl_output.strip():
                yield f"data: {json.dumps({'progress': 0, 'status': 'error', 'error': 'LLM returned empty output'})}\n\n"
                return

            # Event 2: DSL generated, starting rendering
            logger.info(f"[{request_id}] SSE: DSL generated, starting render")
            yield f"data: {json.dumps({'progress': 40, 'status': 'Optimizing geometry...', 'dsl': dsl_output})}\n\n"

            # Step 2: Render diagram (blocking, but we're in async generator)
            from src.infrastructures.celery.tasks import render_diagram_task

            # Run Celery task synchronously in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            render_result = await loop.run_in_executor(
                None,
                lambda: render_diagram_task.apply(
                    kwargs={
                        "task_id": request_id,
                        "dsl": dsl_output,
                        "epochs": 500,
                        "n_tries": 1,
                        "dpi": 150
                    }
                ).result
            )

            # Event 3: Rendering completed
            logger.info(f"[{request_id}] SSE: Render completed")

            # Celery task returns {"image": "...", "status": "...", "task_id": "..."}
            # Map to frontend expected format
            image_data = render_result.get("image") if isinstance(render_result, dict) else None

            yield f"data: {json.dumps({'progress': 100, 'status': 'completed', 'request_id': request_id, 'user_input': user_input, 'dsl': dsl_output, 'image_base64': image_data, 'svg_content': None})}\n\n"

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
