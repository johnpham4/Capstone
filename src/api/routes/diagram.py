from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Optional
from uuid import uuid4
from loguru import logger
import boto3
import json
import io
import traceback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config.settings.base import settings
from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.optimizer import Optimizer
from src.infrastructures.visualization.matplotlib_renderer import MatplotlibDiagramRenderer
from src.models.schemas import RenderRequest, DiagramRequest, DiagramResponse

INSTRUCTION_PROMPT = """Chuyển đổi bài toán hình học tiếng Việt sang Geometry DSL (S-expression syntax).

═══ CÚ PHÁP DSL ═══

1. HÌNH (SHAPES)
   • Tam giác: (triangle (A B C) [type])
     - Thường: (triangle (A B C))
     - Cân: (triangle (A B C) (isosceles A))
     - Vuông: (triangle (A B C) (right B))
     - Vuông cân: (triangle (A B C) (right_isosceles B))
     - Đều: (triangle (A B C) (equilateral))

   • Hình vuông: (square (A B C D))
   CHỈ khai báo (square ...), KHÔNG assert thuộc tính tự nhiên (cạnh bằng nhau, góc vuông, song song)

2. ĐIỂM (POINTS): (define <name> point <construction>)
   - (midpoint B C) - trung điểm
   - (centroid A B C) - trọng tâm
   - (orthocenter A B C) - trực tâm
   - (incenter A B C) - tâm nội tiếp
   - (circumcenter A B C) - tâm ngoại tiếp
   - (projection A (segment B C)) - hình chiếu/đường cao
   - (bisector B A C) - phân giác góc BAC từ đỉnh A
   - (segment A B) - điểm trên đoạn thẳng
   - (line A B) - điểm trên đường thẳng

3. ĐOẠN/ĐƯỜNG:
   - (segment A B) - đoạn thẳng (hữu hạn)
   - (line A B) - đường thẳng (vô hạn)

4. ĐƯỜNG TRÒN: (circle <center> <type>)
   - (incircle A B C) - nội tiếp tam giác
   - (circumcircle A B C) - ngoại tiếp tam giác
   - (incircle A B C D) - nội tiếp hình vuông
   - (circumcircle A B C D) - ngoại tiếp hình vuông
   LUÔN khai báo CẢ tâm (define point) VÀ đường tròn (circle)

5. RÀNG BUỘC:
   - (parallel (segment B C) (segment D E))
   - (perpendicular (segment A B) (segment C D))
   - (angle-equal A B C D E F) - ∠ABC = ∠DEF
   Khai báo segment/line TRƯỚC khi dùng ràng buộc

═══ TỪ KHÓA TIẾNG VIỆT → DSL ═══
- trung điểm → midpoint
- trọng tâm → centroid
- trực tâm → orthocenter
- tâm nội tiếp/đường tròn nội tiếp → incenter/incircle
- tâm ngoại tiếp/đường tròn ngoại tiếp → circumcenter/circumcircle
- đường cao/chân đường cao/hình chiếu → projection
- cân tại → isosceles
- vuông tại → right
- đều → equilateral
- hình vuông → square
- tâm hình vuông → midpoint (của đường chéo)
- đường chéo → segment
- song song → parallel
- vuông góc → perpendicular
- phân giác/đường phân giác -> bisector
- góc bằng nhau/hai góc bằng nhau → angle-equal


═══ QUY TẮC QUAN TRỌNG ═══

1. THỨ TỰ KHAI BÁO:
   ① Hình (triangle/square) - LUÔN TRƯỚC
   ② Define points
   ③ Segments/Lines
   ④ Circles
   ⑤ Constraints (parallel/perpendicular)

2. TRƯỜNG HỢP ĐẶC BIỆT:
   • Trung tuyến AM: (define M point (midpoint B C)) + (segment A M)
   • Đường cao AH: (define H point (projection A (segment B C))) + (segment A H)
   • Phân giác AD: (define D point (bisector B A C)) + (segment A D)
   • Đường thẳng vuông góc qua C: (define H point (projection C (segment A B))) + (line C H)
   • Đường trung bình DE: (define D point (midpoint A B)) + (define E point (midpoint A C)) + (segment D E)
   • Tâm hình vuông O: (define O point (midpoint A C))
   • Đường chéo hình vuông: (segment A C) và/hoặc (segment B D)

3. HÌNH VUÔNG:
   • "Hình vuông ABCD" / "góc vuông" / "cạnh bằng nhau" → CHỈ (square (A B C D))
   • CHỈ thêm khai báo khi đề bài yêu cầu: tâm, đường chéo, trung điểm, đường tròn

4. ĐƯỜNG TRÒN:
   • LUÔN: (define tâm point ...) TRƯỚC → (circle tâm ...) SAU
   • Tam giác: incircle/circumcircle với 3 điểm
   • Hình vuông: incircle/circumcircle với 4 điểm

═══ VÍ DỤ ═══

1. Tam giác cơ bản:
   "Tam giác ABC, M là trung điểm BC"
   → (triangle (A B C))
(define M point (midpoint B C))
(segment A M)

2. Tam giác với đường tròn:
   "Tam giác ABC vuông tại B, đường tròn nội tiếp O"
   → (triangle (A B C) (right B))
(define O point (incenter A B C))
(circle O (incircle A B C))

3. Tam giác cân với đường cao:
   "Tam giác ABC cân tại A, đường cao AH"
   → (triangle (A B C) (isosceles A))
(define H point (projection A (segment B C)))
(segment A H)

4. "Tam giác ABC có AD là đường phân giác của góc A"
   → (triangle (A B C))
(define D point (bisector B A C))
(segment A D)

TRƯỜNG HỢP KHÁC:
"Tam giác ABC có góc BAD bằng góc CAD, M là trung điểm của BC."
   → (triangle (A B C))
(define M point (midpoint B C))
(angle-equal B A D C A D)
(segment A D)

5. Tam giác với song song:
   "Tam giác ABC, D trên AB, E trên AC, BC // DE"
   → (triangle (A B C))
(define D point (segment A B))
(define E point (segment A C))
(segment B C)
(segment D E)
(parallel (segment B C) (segment D E))

6. Hình vuông đơn giản:
   "Hình vuông ABCD" / "Hình vuông ABCD có AB vuông góc BC"
   → (square (A B C D))

7. Hình vuông với đường chéo:
   "Hình vuông ABCD, hai đường chéo AC và BD cắt nhau tại O"
   → (square (A B C D))
(define O point (midpoint A C))
(segment A C)
(segment B D)

8. Hình vuông với đường tròn:
   "Hình vuông ABCD nội tiếp đường tròn"
   → (square (A B C D))
(define O point (midpoint A C))
(circle O (circumcircle A B C D))

9. Tam giác với góc bằng nhau:
   "Tam giác ABC có góc BAD bằng góc CAD"
   → (triangle (A B C))
(define D point (segment B C))
(angle-equal B A D C A D)

Hãy chuyển đổi đề bài sau:
{query}
"""


router = APIRouter()

# Initialize boto3 client for SageMaker
sagemaker_client = boto3.client(
    'sagemaker-runtime',
    region_name=settings.AWS_REGION,
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
)


# ============ API Endpoints ============


@router.post("/api/v1/diagrams/render")
async def render_diagram(request: RenderRequest):
    try:
        request_id = str(uuid4())
        logger.info(f"[{request_id}] Rendering DSL: {request.dsl[:100]}...")

        # Parse DSL
        dsl_lines = request.dsl.split('\n') if '\n' in request.dsl else [request.dsl]
        dsl_lines = [line.strip() for line in dsl_lines if line.strip()]

        builder = DiagramBuilder(dsl_lines)
        logger.info(f"[{request_id}] Built diagram with {len(builder.points)} points")

        # Optimize diagram layout
        opts = {
            'epochs': request.epochs,
            'n_tries': request.n_tries,
            'eps': 1e-6,
            'seed': 42
        }
        optimizer = Optimizer(builder.instructions, opts, verbosity=False)
        diagram = optimizer.solve()

        logger.info(f"[{request_id}] Optimization complete")

        # Render to image
        renderer = MatplotlibDiagramRenderer()
        fig, ax = renderer.render(diagram, save=False, show=False)

        # Add title if provided
        if request.title:
            fig.suptitle(request.title, fontsize=10, wrap=True)

        # Save to BytesIO buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=request.dpi, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        logger.info(f"[{request_id}] Image generated successfully")

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=diagram_{request_id}.png",
                "X-Request-ID": request_id
            }
        )

    except Exception as e:
        logger.exception(f"Render failed: {e}")
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "trace": error_trace
            }
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


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test endpoint is accessible
        # You can add a lightweight health check call here
        return {
            "status": "healthy",
            "service": "GeoUni Diagram Generation API",
            "endpoint": settings.SAGEMAKER_ENDPOINT_INFERENCE
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@router.post("/api/v1/diagrams/generate-and-render")
async def generate_and_render_diagram(request: DiagramRequest):
    try:
        request_id = str(uuid4())
        logger.info(f"[{request_id}] Full pipeline request: {request.user_input}")

        # Step 1: Generate DSL from LLM
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

        # Extract DSL from response
        dsl_output = ""
        if 'choices' in result and len(result['choices']) > 0:
            choice = result['choices'][0]
            if 'message' in choice:
                dsl_output = choice['message'].get('content', '')
            elif 'text' in choice:
                dsl_output = choice['text']
        else:
            dsl_output = result.get("generated_text", result.get("text", ""))

        logger.info(f"[{request_id}] Generated DSL: {dsl_output[:100]}...")

        if not dsl_output.strip():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM returned empty output"
            )

        # Step 2: Parse and render DSL
        dsl_lines = dsl_output.split('\n') if '\n' in dsl_output else [dsl_output]
        dsl_lines = [line.strip() for line in dsl_lines if line.strip()]

        builder = DiagramBuilder(dsl_lines)
        logger.info(f"[{request_id}] Built diagram with {len(builder.points)} points")

        # Step 3: Optimize
        opts = {'epochs': 2000, 'n_tries': 1, 'eps': 1e-6, 'seed': 42}
        optimizer = Optimizer(builder.instructions, opts, verbosity=False)
        diagram = optimizer.solve()

        # Step 4: Render
        renderer = MatplotlibDiagramRenderer()
        fig, ax = renderer.render(diagram, save=False, show=False)
        fig.suptitle(request.user_input, fontsize=10, wrap=True)

        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        logger.info(f"[{request_id}] Full pipeline complete")

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=diagram_{request_id}.png",
                "X-Request-ID": request_id,
                "X-Generated-DSL": dsl_output[:200]  # First 200 chars of DSL
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Full pipeline failed: {e}")
        error_trace = traceback.format_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": str(e),
                "trace": error_trace
            }
        )

