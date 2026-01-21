from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from uuid import uuid4
from pathlib import Path
from loguru import logger

from llm_engineering.infrastructures.db.rabbitmq import RabbitMQPublisher
from llm_engineering.infrastructures.db.processing_request_repository import processing_request_repository
from llm_engineering.domains.processing_request import ProcessingRequest, RequestStatus
from llm_engineering.domains.events import UserInputReceived


app = FastAPI(
    title="GeoUni Diagram Generation API",
    description="Event-driven API for geometry diagram generation",
    version="1.0.0"
)

# Initialize RabbitMQ publisher
publisher = RabbitMQPublisher()
INPUT_QUEUE = "model_processing_queue"


class DiagramRequest(BaseModel):
    """Request to generate a geometry diagram"""
    user_input: str = Field(..., description="User's geometry problem or DSL commands")
    problem_text: Optional[str] = Field(None, description="Optional problem description")


class DiagramResponse(BaseModel):
    """Response containing request tracking information"""
    request_id: str
    status: str
    message: str


class DiagramStatusResponse(BaseModel):
    """Response containing request status and results"""
    request_id: str
    status: str
    user_input: str
    model_output: Optional[str] = None
    diagram_path: Optional[str] = None
    diagram_points: Optional[dict] = None
    error_message: Optional[str] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize queues on startup"""
    try:
        publisher.connect()
        publisher.declare_queue(INPUT_QUEUE)
        logger.info("API server started successfully")
    except Exception as e:
        logger.exception(f"Failed to start API server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    publisher.close()
    logger.info("API server shutdown")


@app.post("/api/v1/diagrams", response_model=DiagramResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_diagram_request(request: DiagramRequest):
    """
    Submit a geometry diagram generation request.
    Returns a request_id for tracking the processing status.
    """
    try:
        # Generate unique request ID
        request_id = str(uuid4())

        # Create processing request in database
        processing_request = ProcessingRequest(
            request_id=request_id,
            status=RequestStatus.PENDING,
            user_input=request.user_input,
            problem_text=request.problem_text
        )

        processing_request_repository.create(processing_request)

        # Publish event to input processing queue
        event = UserInputReceived(
            request_id=request_id,
            user_input=request.user_input,
            problem_text=request.problem_text
        )

        publisher.publish(INPUT_QUEUE, event.model_dump())

        logger.info(f"Created diagram request: {request_id}")

        return DiagramResponse(
            request_id=request_id,
            status="processing",
            message="Request received and queued for processing"
        )

    except Exception as e:
        logger.exception(f"Failed to create diagram request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create request: {str(e)}"
        )


@app.get("/api/v1/diagrams/{request_id}", response_model=DiagramStatusResponse)
async def get_diagram_status(request_id: str):
    """
    Get the status and results of a diagram generation request.
    """
    try:
        # Fetch request from database
        processing_request = processing_request_repository.get_by_id(request_id)

        if not processing_request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Request {request_id} not found"
            )

        return DiagramStatusResponse(
            request_id=processing_request.request_id,
            status=processing_request.status,
            user_input=processing_request.user_input,
            model_output=processing_request.model_output,
            diagram_path=processing_request.diagram_path,
            diagram_points=processing_request.diagram_points,
            error_message=processing_request.error_message,
            created_at=processing_request.created_at.isoformat(),
            updated_at=processing_request.updated_at.isoformat(),
            completed_at=processing_request.completed_at.isoformat() if processing_request.completed_at else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get diagram status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get status: {str(e)}"
        )


@app.get("/api/v1/diagrams/{request_id}/image")
async def get_diagram_image(request_id: str):
    """
    Download the generated diagram image.
    """
    try:
        # Fetch request from database
        processing_request = processing_request_repository.get_by_id(request_id)

        if not processing_request:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Request {request_id} not found"
            )

        if processing_request.status != RequestStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Diagram not ready yet. Current status: {processing_request.status}"
            )

        if not processing_request.diagram_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Diagram file not found"
            )

        diagram_path = Path(processing_request.diagram_path)

        if not diagram_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Diagram file does not exist on server"
            )

        return FileResponse(
            path=str(diagram_path),
            media_type="image/png",
            filename=f"diagram_{request_id}.png"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get diagram image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get image: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "GeoUni Diagram Generation API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
