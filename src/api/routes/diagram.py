<<<<<<< HEAD
from fastapi import APIRouter, HTTPException, status
from uuid import uuid4
from loguru import logger
import boto3
import json

from .prompt import INSTRUCTION_PROMPT
from src.config.settings.base import settings
from src.models.schemas import DiagramRequest, DiagramResponse


router = APIRouter()

# Initialize boto3 client for SageMaker
sagemaker_client = boto3.client(
    'sagemaker-runtime',
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
)


@router.post("/api/v1/diagrams/generate", response_model=DiagramResponse)
async def generate_diagram(request: DiagramRequest):
    """
    Generate geometry diagram DSL from user input.
    """
    try:
        request_id = str(uuid4())
        logger.info(f"[{request_id}] Request: {request.user_input}")

        # Payload cho vLLM endpoint (OpenAI-compatible format)
        full_prompt = INSTRUCTION_PROMPT.format(query=request.user_input)
        payload = {
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": 0.9,
            "top_k": 50
        }

        # Call SageMaker endpoint
        response = sagemaker_client.invoke_endpoint(
            EndpointName=settings.SAGEMAKER_ENDPOINT_INFERENCE,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        # Parse response (vLLM returns OpenAI-compatible format)
        result = json.loads(response['Body'].read().decode('utf-8'))
        # Extract content from choices array
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice:
                model_output = choice['message'].get('content', '')
            elif 'text' in choice:
                model_output = choice['text']
            else:
                model_output = ''
        else:
            model_output = result.get("generated_text", result.get("text", ""))

        logger.info(f"[{request_id}] Output: {model_output[:100]}...")

        return DiagramResponse(
            request_id=request_id,
            user_input=request.user_input,
            model_output=model_output,
            status="completed"
        )

    except Exception as e:
        logger.exception(f"Failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/api/v1/chat", response_model=dict)
async def chat(request: DiagramRequest):
    """
    Simple chat endpoint for testing chatbot inference.
    """
    try:
        logger.info(f"Chat: {request.user_input}")

        # Payload cho vLLM endpoint (OpenAI-compatible format)
        full_prompt = INSTRUCTION_PROMPT.format(query=request.user_input)
        payload = {
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": 0.9,
            "top_k": 50
        }

        response = sagemaker_client.invoke_endpoint(
            EndpointName=settings.SAGEMAKER_ENDPOINT_INFERENCE,
            ContentType='application/json',
            Body=json.dumps(payload)
        )

        result = json.loads(response['Body'].read().decode('utf-8'))

        # Extract content from OpenAI-compatible response
        output = ""
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice:
                output = choice['message'].get('content', '')
            elif 'text' in choice:
                output = choice['text']
        else:
            output = result.get("generated_text", result.get("text", ""))

        return {
            "input": request.user_input,
            "output": output,
            "model": settings.HF_MODEL_ID
        }

    except Exception as e:
        logger.exception(f"Chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/api/v1/model/info")
async def get_model_info():
    """Get current model information"""
    return {
        "model_id": settings.HF_MODEL_ID,
        "endpoint": settings.SAGEMAKER_ENDPOINT_INFERENCE,
        "instance_type": settings.GPU_INSTANCE_TYPE,
        "status": "ready"
    }


@router.post("/api/v1/diagrams/full-pipeline")
async def full_pipeline(request: DiagramRequest):
    try:
        request_id = str(uuid4())
        logger.info(f"[{request_id}] Full pipeline: {request.user_input[:50]}...")

        # Step 1: Generate DSL via SageMaker (async I/O)
        full_prompt = INSTRUCTION_PROMPT.format(query=request.user_input)
        payload = {
            "messages": [{"role": "user", "content": full_prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
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

        logger.info(f"[{request_id}] DSL generated: {dsl_output[:100]}...")

        if not dsl_output.strip():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM returned empty output"
            )

        # Step 2: Queue render task to Celery (CPU-intensive work)
        from src.infrastructures.celery.tasks import render_diagram_task

        celery_task = render_diagram_task.apply_async(
            kwargs={
                "task_id": request_id,
                "dsl": dsl_output,
                "epochs": 500,
                "n_tries": 1,
                "dpi": 150
            }
        )

        logger.info(f"[{request_id}] Queued render task: {celery_task.id}")

        return {
            "request_id": request_id,
            "user_input": request.user_input,
            "dsl": dsl_output,
            "celery_task_id": celery_task.id,
            "status": "rendering",
            "message": "DSL generated, image rendering in background. Poll /api/tasks/status/{celery_task_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Full pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

=======
import json
import time
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse, Response
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.rate_limiter import rate_limit_diagram
from src.infrastructures.database.session import get_db
from src.models.dto.user import User
from src.prompts import DSL_INFERENCE_INSTRUCTION
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
>>>>>>> minh-re
