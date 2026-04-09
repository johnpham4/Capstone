import json

from loguru import logger
from src.infrastructures.celery.config import celery_app
from src.services.diagram.generation import DiagramService


diagram_service = DiagramService()


@celery_app.task(bind=True, name="render_diagram", store_errors_even_if_ignored=True)
def render_diagram_task(self, task_id: str, dsl: str, epochs: int = 500, n_tries: int = 1, dpi: int = 150):
    """Worker task: Render diagram from DSL (CPU-intensive)"""
    logger.info(f"[Worker] Rendering diagram for task {task_id}")

    try:
        result = diagram_service.generate_and_render(
            task_id=task_id,
            dsl=dsl,
            epochs=epochs,
            n_tries=n_tries,
            dpi=dpi,
        )

        if result.get("status") != "success":
            raise RuntimeError(
                json.dumps(
                    {
                        "error_code": result.get("error_code", "DIAGRAM_GENERATION_ERROR"),
                        "message": result.get("error", "Diagram generation failed"),
                    }
                )
            )

        logger.info(f"[Worker {task_id}] Completed")

        return {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "completed",
            "result": result,
        }

    except Exception as e:
        logger.error(f"[Worker {task_id}] Failed: {e}")
        error_payload = {
            "error_code": "DIAGRAM_TASK_ERROR",
            "message": str(e),
        }

        try:
            parsed = json.loads(str(e))
            if isinstance(parsed, dict):
                error_payload["error_code"] = str(parsed.get("error_code", error_payload["error_code"]))
                error_payload["message"] = str(parsed.get("message", error_payload["message"]))
        except Exception:
            pass

        raise RuntimeError(json.dumps(error_payload))

