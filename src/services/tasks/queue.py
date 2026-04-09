import json
from uuid import uuid4

from celery import current_app
from celery.result import AsyncResult

from src.config.settings.settings import settings
from src.infrastructures.celery.tasks import render_diagram_task
from src.services.tasks.ports import TaskQueuePort


class CeleryTaskQueueService(TaskQueuePort):
    STATUS_BY_STATE = {
        "PENDING": "queued",
        "STARTED": "running",
        "SUCCESS": "completed",
        "FAILURE": "failed",
        "RETRY": "running",
        "REVOKED": "failed",
    }

    def _enqueue_diagram_render(self, dsl: str, epochs: int, n_tries: int, dpi: int) -> dict:
        task_id = str(uuid4())
        celery_task = render_diagram_task.apply_async(
            kwargs={
                "task_id": task_id,
                "dsl": dsl,
                "epochs": epochs,
                "n_tries": n_tries,
                "dpi": dpi,
            },
            queue=settings.DIAGRAM_QUEUE_NAME,
        )
        return {
            "task_id": task_id,
            "celery_task_id": celery_task.id,
            "status": "queued",
            "queue": settings.DIAGRAM_QUEUE_NAME,
        }

    def queue_diagram_render(self, dsl: str, epochs: int, n_tries: int, dpi: int) -> dict:
        return self._enqueue_diagram_render(
            dsl=dsl,
            epochs=epochs,
            n_tries=n_tries,
            dpi=dpi,
        )

    def get_task_status(self, celery_task_id: str) -> dict:
        result = AsyncResult(celery_task_id)
        normalized_status = self.STATUS_BY_STATE.get(result.state, "running")
        payload = {
            "task_id": celery_task_id,
            "celery_task_id": celery_task_id,
            "status": normalized_status,
            "raw_state": result.state,
            "progress": None,
            "result": None,
            "error_code": None,
            "error": None,
        }

        if result.state == "PENDING":
            payload["progress"] = 0
        elif result.state == "STARTED":
            payload["progress"] = 50
        elif result.state == "SUCCESS":
            payload["progress"] = 100
            task_payload = result.result if isinstance(result.result, dict) else {"value": result.result}
            payload["task_id"] = task_payload.get("task_id", payload["task_id"])
            payload["result"] = task_payload.get("result", task_payload)
        elif result.state == "FAILURE":
            payload["error"] = str(result.info)
            try:
                parsed_error = json.loads(str(result.info))
                if isinstance(parsed_error, dict):
                    payload["error_code"] = parsed_error.get("error_code")
                    payload["error"] = parsed_error.get("message", payload["error"])
            except Exception:
                payload["error_code"] = "DIAGRAM_TASK_ERROR"

        return payload

    def get_workers_status(self) -> dict:
        inspect = current_app.control.inspect()
        stats = inspect.stats()
        active = inspect.active()
        registered = inspect.registered()

        return {
            "workers": stats if stats else {},
            "active_tasks": active if active else {},
            "registered_tasks": registered if registered else {},
            "status": "online" if stats else "offline",
        }

    def get_active_tasks(self) -> dict:
        inspect = current_app.control.inspect()
        active = inspect.active()

        if not active:
            return {"active_tasks": [], "count": 0}

        tasks = []
        for worker, task_list in active.items():
            for task in task_list:
                tasks.append(
                    {
                        "worker": worker,
                        "task_id": task.get("id"),
                        "name": task.get("name"),
                        "args": task.get("args"),
                        "time_start": task.get("time_start"),
                    }
                )

        return {"active_tasks": tasks, "count": len(tasks)}



TaskQueueService = CeleryTaskQueueService
