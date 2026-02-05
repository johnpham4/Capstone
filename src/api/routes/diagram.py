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

