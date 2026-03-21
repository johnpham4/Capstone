from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    error: str
    detail: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)


class SuccessResponse(BaseModel):
    """Standard success response schema."""
    message: str
    request_id: Optional[str] = None
    data: Optional[dict] = None

