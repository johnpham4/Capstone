from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger
import json
import asyncio

router = APIRouter()

# Store active WebSocket connections
active_connections: dict[str, WebSocket] = {}


@router.websocket("/ws/tasks/{task_id}")
async def websocket_task_status(websocket: WebSocket, task_id: str):
    await websocket.accept()
    active_connections[task_id] = websocket
    logger.info(f"[WS] Client connected for task {task_id}")

    try:
        from celery.result import AsyncResult

        while True:
            result = AsyncResult(task_id)

            status_data = {
                "task_id": task_id,
                "status": result.state,
                "timestamp": asyncio.get_event_loop().time()
            }

            if result.state == 'PENDING':
                status_data["progress"] = 0
                status_data["message"] = "Task queued, waiting for worker..."
            elif result.state == 'STARTED':
                status_data["progress"] = 50
                status_data["message"] = "Processing..."
            elif result.state == 'SUCCESS':
                status_data["progress"] = 100
                status_data["result"] = result.result
                status_data["message"] = "Completed!"
                await websocket.send_json(status_data)
                logger.info(f"[WS] Task {task_id} completed, closing connection")
                break
            elif result.state == 'FAILURE':
                status_data["error"] = str(result.info)
                status_data["message"] = "Failed!"
                await websocket.send_json(status_data)
                break

            await websocket.send_json(status_data)
            await asyncio.sleep(1)  # Check every 1 second

    except WebSocketDisconnect:
        logger.info(f"[WS] Client disconnected from task {task_id}")
    except Exception as e:
        logger.error(f"[WS] Error: {e}")
    finally:
        if task_id in active_connections:
            del active_connections[task_id]
        await websocket.close()


async def notify_task_completion(task_id: str, result: dict):
    """
    Called by Celery task to notify WebSocket clients.
    This is a callback mechanism.
    """
    if task_id in active_connections:
        websocket = active_connections[task_id]
        try:
            await websocket.send_json({
                "task_id": task_id,
                "status": "SUCCESS",
                "result": result,
                "message": "Task completed!"
            })
            logger.info(f"[WS] Notified client for task {task_id}")
        except Exception as e:
            logger.error(f"[WS] Failed to notify: {e}")
