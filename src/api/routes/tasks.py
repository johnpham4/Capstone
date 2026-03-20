<<<<<<< HEAD
from fastapi import APIRouter, HTTPException
from uuid import uuid4
from loguru import logger
from pydantic import BaseModel, Field
from typing import Optional

from src.infrastructures.celery.tasks import render_diagram_task

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


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
    task_id = str(uuid4())
    logger.info(f"[API] Queuing render task: {task_id}")

    celery_task = render_diagram_task.apply_async(
        kwargs={
            "task_id": task_id,
            "dsl": request.dsl,
            "epochs": request.epochs,
            "n_tries": request.n_tries,
            "dpi": request.dpi
        }
    )

    return TaskResponse(
        task_id=task_id,
        celery_task_id=celery_task.id,
        status="queued"
    )


@router.get("/status/{celery_task_id}", response_model=TaskStatusResponse)
async def get_task_status(celery_task_id: str):
    from celery.result import AsyncResult

    result = AsyncResult(celery_task_id)

    response = TaskStatusResponse(
        task_id=celery_task_id,
        status=result.state
    )

    if result.state == 'PENDING':
        response.progress = 0
    elif result.state == 'STARTED':
        response.progress = 50
    elif result.state == 'SUCCESS':
        response.progress = 100
        response.result = result.result
    elif result.state == 'FAILURE':
        response.error = str(result.info)

    return response


@router.get("/workers/status")
async def get_workers_status():
    """Get status of all Celery workers"""
    from celery import current_app

    inspect = current_app.control.inspect()

    stats = inspect.stats()
    active = inspect.active()
    registered = inspect.registered()

    return {
        "workers": stats if stats else {},
        "active_tasks": active if active else {},
        "registered_tasks": registered if registered else {},
        "status": "online" if stats else "offline"
    }


@router.get("/tasks/active")
async def get_active_tasks():
    """Get all active (running) tasks"""
    from celery import current_app

    inspect = current_app.control.inspect()
    active = inspect.active()

    if not active:
        return {"active_tasks": [], "count": 0}

    tasks = []
    for worker, task_list in active.items():
        for task in task_list:
            tasks.append({
                "worker": worker,
                "task_id": task.get('id'),
                "name": task.get('name'),
                "args": task.get('args'),
                "time_start": task.get('time_start')
            })

    return {"active_tasks": tasks, "count": len(tasks)}
=======
from fastapi import APIRouter
from loguru import logger

from src.models.dto.task import RenderTaskRequest, TaskResponse, TaskStatusResponse
from src.services.tasks.queue import TaskQueueService

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])
task_queue_service = TaskQueueService()


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
>>>>>>> 6cf03dda8dad8bb8fa1226b8b4e9166c3f287527
