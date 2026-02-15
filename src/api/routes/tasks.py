from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel, Field
from typing import Optional

from src.services.tasks.task_queue_service import TaskQueueService

router = APIRouter(prefix="/api/tasks", tags=["tasks"])
task_queue_service = TaskQueueService()


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


@router.post("/diagrams/render", response_model=TaskResponse)
async def queue_diagram_render(request: RenderTaskRequest):
    """
    Queue diagram rendering task to Celery workers.
    Returns immediately with task_id for status checking.
    """
    task_data = task_queue_service.queue_diagram_render(
        dsl=request.dsl,
        epochs=request.epochs,
        n_tries=request.n_tries,
        dpi=request.dpi,
    )
    logger.info(f"[API] Queued render task: {task_data['task_id']}")
    return TaskResponse(**task_data)


@router.get("/status/{celery_task_id}", response_model=TaskStatusResponse)
async def get_task_status(celery_task_id: str):
    return TaskStatusResponse(**task_queue_service.get_task_status(celery_task_id))


@router.get("/workers/status")
async def get_workers_status():
    """Get status of all Celery workers"""
    return task_queue_service.get_workers_status()


@router.get("/tasks/active")
async def get_active_tasks():
    """Get all active (running) tasks"""
    return task_queue_service.get_active_tasks()
