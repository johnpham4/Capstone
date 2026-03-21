"""Task queue DTOs."""

from pydantic import BaseModel, Field
from typing import Optional


class RenderTaskRequest(BaseModel):
    dsl: str
    epochs: int = Field(default=500, ge=100, le=2000)
    n_tries: int = Field(default=1, ge=1, le=5)
    dpi: int = Field(default=150, ge=72, le=300)


class TaskResponse(BaseModel):
    task_id: str
    celery_task_id: str
    status: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # PENDING, STARTED, SUCCESS, FAILURE
    progress: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[str] = None

