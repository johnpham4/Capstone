import json

from loguru import logger
from src.infrastructures.celery.config import celery_app
from src.services.diagram.generation import DiagramService


diagram_service = DiagramService()


@celery_app.task(bind=True, name="render_diagram", store_errors_even_if_ignored=True)
def render_diagram_task(self, task_id: str, dsl: str, epochs: int = 2000, dpi: int = 150):
    """Worker task: Render diagram from DSL (CPU-intensive)"""
    logger.info(f"[Worker] Rendering diagram for task {task_id}")
    logger.info(f"[Worker] DSL input:\n{dsl}")

    try:
        result = diagram_service.generate_and_render(
            task_id=task_id,
            dsl=dsl,
            epochs=epochs,
            dpi=dpi,
        )

        if result.get("status") != "success":
            error_code = result.get("error_code", "DIAGRAM_GENERATION_ERROR")
            message = result.get("error", "Diagram generation failed")
            raise RuntimeError(json.dumps({"error_code": error_code, "message": message}))

        logger.info(f"[Worker {task_id}] Completed")

        return {
            "task_id": task_id,
            "celery_task_id": self.request.id,
            "status": "completed",
            "result": result,
        }

    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"[Worker {task_id}] Failed: {e}")
        raise RuntimeError(json.dumps({
            "error_code": "DIAGRAM_TASK_ERROR",
            "message": str(e),
        }))

