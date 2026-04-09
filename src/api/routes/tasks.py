import asyncio
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from loguru import logger

from src.models.dto.task import RenderTaskRequest, TaskResponse, TaskStatusResponse
from src.services.tasks.queue import TaskQueueService

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])
task_queue_service = TaskQueueService()


@router.post("/diagrams/render", response_model=TaskResponse)
async def queue_diagram_render(request: RenderTaskRequest):
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


@router.get("/status/stream/{celery_task_id}")
async def stream_task_status(celery_task_id: str):
    async def event_generator():
        terminal_statuses = {"completed", "failed"}

        while True:
            status_payload = task_queue_service.get_task_status(celery_task_id)
            normalized = status_payload.get("status", "running")
            status_json = json.dumps(status_payload, ensure_ascii=True)

            yield f"event: task_status\ndata: {status_json}\n\n"

            if normalized in terminal_statuses:
                yield f"event: task_done\ndata: {status_json}\n\n"
                break

            await asyncio.sleep(1.0)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_generator(), media_type="text/event-stream", headers=headers)


@router.get("/workers/status")
async def get_workers_status():
    return task_queue_service.get_workers_status()


@router.get("/tasks/active")
async def get_active_tasks():
    return task_queue_service.get_active_tasks()
