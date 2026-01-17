from enum import StrEnum
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class RequestStatus(StrEnum):
    """Status of processing request"""
    PENDING = "pending"
    PROCESSING_INPUT = "processing_input"
    PROCESSING_MODEL = "processing_model"
    GENERATING_DIAGRAM = "generating_diagram"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingRequest(BaseModel):
    """Domain model for tracking processing requests"""
    request_id: str
    status: RequestStatus = RequestStatus.PENDING
    user_input: str
    problem_text: Optional[str] = None

    # Processing results
    model_output: Optional[str] = None
    dsl_commands: Optional[list[str]] = None
    diagram_path: Optional[str] = None
    diagram_points: Optional[Dict[str, Any]] = None

    # Metadata
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


    def update_status(self, status: RequestStatus, error_message: Optional[str] = None):
        """Update request status"""
        self.status = status
        self.updated_at = datetime.utcnow()
        if error_message:
            self.error_message = error_message
        if status == RequestStatus.COMPLETED or status == RequestStatus.FAILED:
            self.completed_at = datetime.utcnow()
