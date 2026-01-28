from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from uuid import uuid4
from loguru import logger
import json

from llm_src.model.inference.inference import LLMInferenceSagemakerEndpoint
from llm_src.settings import settings

app = FastAPI(
    title="GeoUni Chat Inference API",
    description="FastAPI endpoint for chat inference with AWS SageMaker + vLLM",
    version="1.0.0"
)


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Chat inference request"""
    messages: List[ChatMessage] = Field(
        ...,
        description="List of chat messages in conversation format"
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for sampling (0.0 to 2.0)"
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling parameter"
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k sampling parameter"
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Whether to stream the response"
    )


class SimpleChatRequest(BaseModel):
    """Simple chat inference request with single prompt"""
    prompt: str = Field(..., description="Single prompt text for inference")
    max_new_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    system_prompt: Optional[str] = Field(
        default=None,
        description="Optional system prompt"
    )


class ChatResponse(BaseModel):
    """Chat inference response"""
    request_id: str
    generated_text: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class ChatStreamResponse(BaseModel):
    """Streaming chat response chunk"""
    request_id: str
    token: str
    finish_reason: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    endpoint_name: str
    message: str


# Initialize SageMaker inference client (lazy loading)
_inference_client: Optional[LLMInferenceSagemakerEndpoint] = None


def get_inference_client() -> LLMInferenceSagemakerEndpoint:
    """
    Get or create the SageMaker inference client.
    """
    global _inference_client

    if _inference_client is None:
        if not settings.SAGEMAKER_ENDPOINT_INFERENCE:
            raise ValueError(
                "SAGEMAKER_ENDPOINT_INFERENCE is not configured in settings. "
                "Please set it in your .env file."
            )

        _inference_client = LLMInferenceSagemakerEndpoint(
            endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE
        )
        logger.info(f"Initialized SageMaker inference client with endpoint: {settings.SAGEMAKER_ENDPOINT_INFERENCE}")

    return _inference_client


def format_messages_to_prompt(messages: List[ChatMessage]) -> str:
    """
    Format chat messages into a single prompt string.
    This uses a simple format - you may want to use a proper chat template.
    """
    formatted = ""
    for msg in messages:
        if msg.role == "system":
            formatted += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            formatted += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            formatted += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"

    # Add assistant prefix for generation
    formatted += "<|im_start|>assistant\n"
    return formatted


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check"""
    return HealthResponse(
        status="healthy",
        endpoint_name=settings.SAGEMAKER_ENDPOINT_INFERENCE,
        message="GeoUni Chat Inference API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Try to get the inference client to verify configuration
        client = get_inference_client()
        return HealthResponse(
            status="healthy",
            endpoint_name=client.endpoint_name,
            message="SageMaker endpoint is configured and ready"
        )
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_inference(request: ChatRequest):
    """
    Main chat inference endpoint with conversational format.

    Example request:
    ```json
    {
        "messages": [
            {"role": "system", "content": "You are a helpful geometry assistant."},
            {"role": "user", "content": "Convert this problem to DSL: Triangle ABC vuông tại B"}
        ],
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9
    }
    ```
    """
    request_id = str(uuid4())

    try:
        # Get inference client
        client = get_inference_client()

        # Format messages to prompt
        prompt = format_messages_to_prompt(request.messages)

        # Prepare parameters
        parameters: Dict[str, Any] = {}

        if request.max_new_tokens is not None:
            parameters["max_new_tokens"] = request.max_new_tokens
        else:
            parameters["max_new_tokens"] = settings.MAX_NEW_TOKENS_INFERENCE

        if request.temperature is not None:
            parameters["temperature"] = request.temperature
        else:
            parameters["temperature"] = settings.TEMPERATURE_INFERENCE

        if request.top_p is not None:
            parameters["top_p"] = request.top_p
        else:
            parameters["top_p"] = settings.TOP_P_INFERENCE

        if request.top_k is not None:
            parameters["top_k"] = request.top_k

        parameters["return_full_text"] = False

        # Set payload and perform inference
        client.set_payload(inputs=prompt, parameters=parameters)

        logger.info(f"[{request_id}] Sending inference request to SageMaker endpoint")
        logger.debug(f"[{request_id}] Prompt: {prompt[:200]}...")

        response = client.inference()

        # Parse response based on vLLM output format
        # vLLM typically returns: [{"generated_text": "...", "finish_reason": "stop"}]
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "")
            finish_reason = response[0].get("finish_reason")
            usage = response[0].get("usage") if "usage" in response[0] else None
        elif isinstance(response, dict):
            generated_text = response.get("generated_text", "")
            finish_reason = response.get("finish_reason")
            usage = response.get("usage")
        else:
            generated_text = str(response)
            finish_reason = None
            usage = None

        logger.info(f"[{request_id}] Inference completed successfully")
        logger.debug(f"[{request_id}] Generated: {generated_text[:200]}...")

        return ChatResponse(
            request_id=request_id,
            generated_text=generated_text,
            finish_reason=finish_reason,
            usage=usage
        )

    except Exception as e:
        logger.exception(f"[{request_id}] Chat inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/api/v1/chat/simple", response_model=ChatResponse)
async def simple_chat_inference(request: SimpleChatRequest):
    """
    Simplified chat inference endpoint with single prompt.

    Example request:
    ```json
    {
        "prompt": "Chuyển đổi bài toán: Tam giác ABC vuông tại B, M là trung điểm BC",
        "max_new_tokens": 512,
        "temperature": 0.7,
        "system_prompt": "You are a geometry DSL converter."
    }
    ```
    """
    request_id = str(uuid4())

    try:
        # Get inference client
        client = get_inference_client()

        # Build messages
        messages = []

        if request.system_prompt:
            messages.append(ChatMessage(role="system", content=request.system_prompt))

        messages.append(ChatMessage(role="user", content=request.prompt))

        # Format to prompt
        prompt = format_messages_to_prompt(messages)

        # Prepare parameters
        parameters: Dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens or settings.MAX_NEW_TOKENS_INFERENCE,
            "temperature": request.temperature or settings.TEMPERATURE_INFERENCE,
            "top_p": request.top_p or settings.TOP_P_INFERENCE,
            "return_full_text": False,
        }

        # Set payload and perform inference
        client.set_payload(inputs=prompt, parameters=parameters)

        logger.info(f"[{request_id}] Sending simple inference request")
        logger.debug(f"[{request_id}] Prompt: {request.prompt[:200]}...")

        response = client.inference()

        # Parse response
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "")
            finish_reason = response[0].get("finish_reason")
            usage = response[0].get("usage") if "usage" in response[0] else None
        elif isinstance(response, dict):
            generated_text = response.get("generated_text", "")
            finish_reason = response.get("finish_reason")
            usage = response.get("usage")
        else:
            generated_text = str(response)
            finish_reason = None
            usage = None

        logger.info(f"[{request_id}] Simple inference completed")

        return ChatResponse(
            request_id=request_id,
            generated_text=generated_text,
            finish_reason=finish_reason,
            usage=usage
        )

    except Exception as e:
        logger.exception(f"[{request_id}] Simple chat inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@app.post("/api/v1/chat/geometry", response_model=ChatResponse)
async def geometry_chat_inference(request: SimpleChatRequest):
    """
    Specialized endpoint for geometry problem to DSL conversion.
    Automatically includes the geometry instruction prompt.

    Example request:
    ```json
    {
        "prompt": "Tam giác ABC vuông tại B, M là trung điểm của BC. Tính AM.",
        "max_new_tokens": 512,
        "temperature": 0.3
    }
    ```
    """
    request_id = str(uuid4())

    try:
        # Get inference client
        client = get_inference_client()

        # Use the geometry instruction prompt from finetuning
        GEOMETRY_SYSTEM_PROMPT = """Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL (S-expression syntax).

═══ CÚ PHÁP DSL ═══

1. HÌNH (SHAPES)
   • Tam giác: (triangle (A B C) [type])
     - Thường: (triangle (A B C))
     - Cân: (triangle (A B C) (isosceles A))
     - Vuông: (triangle (A B C) (right B))
     - Vuông cân: (triangle (A B C) (right_isosceles B))
     - Đều: (triangle (A B C) (equilateral))

   • Hình vuông: (square (A B C D))

2. ĐIỂM (POINTS): (define <name> point <construction>)
   - (midpoint B C) - trung điểm
   - (centroid A B C) - trọng tâm
   - (orthocenter A B C) - trực tâm
   - (incenter A B C) - tâm nội tiếp
   - (circumcenter A B C) - tâm ngoại tiếp
   - (projection A (segment B C)) - hình chiếu

3. RÀNG BUỘC:
   - (parallel (segment B C) (segment D E))
   - (perpendicular (segment A B) (segment C D))
   - (angle-equal A B C D E F)

Hãy chuyển đổi bài toán sau sang DSL:"""

        # Build messages with geometry-specific system prompt
        messages = [
            ChatMessage(role="system", content=GEOMETRY_SYSTEM_PROMPT),
            ChatMessage(role="user", content=request.prompt)
        ]

        # Format to prompt
        prompt = format_messages_to_prompt(messages)

        # Prepare parameters - use lower temperature for more deterministic DSL output
        parameters: Dict[str, Any] = {
            "max_new_tokens": request.max_new_tokens or 512,
            "temperature": request.temperature or 0.3,  # Lower temperature for structured output
            "top_p": request.top_p or 0.9,
            "return_full_text": False,
        }

        # Set payload and perform inference
        client.set_payload(inputs=prompt, parameters=parameters)

        logger.info(f"[{request_id}] Sending geometry DSL conversion request")
        logger.debug(f"[{request_id}] Problem: {request.prompt}")

        response = client.inference()

        # Parse response
        if isinstance(response, list) and len(response) > 0:
            generated_text = response[0].get("generated_text", "")
            finish_reason = response[0].get("finish_reason")
            usage = response[0].get("usage") if "usage" in response[0] else None
        elif isinstance(response, dict):
            generated_text = response.get("generated_text", "")
            finish_reason = response.get("finish_reason")
            usage = response.get("usage")
        else:
            generated_text = str(response)
            finish_reason = None
            usage = None

        logger.info(f"[{request_id}] Geometry DSL conversion completed")
        logger.debug(f"[{request_id}] DSL output: {generated_text}")

        return ChatResponse(
            request_id=request_id,
            generated_text=generated_text,
            finish_reason=finish_reason,
            usage=usage
        )

    except Exception as e:
        logger.exception(f"[{request_id}] Geometry chat inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
