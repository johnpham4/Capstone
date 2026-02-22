from loguru import logger
from src.infrastructures.celery.config import celery_app


@celery_app.task(bind=True, name="render_diagram")
def render_diagram_task(self, task_id: str, dsl: str, epochs: int = 500, n_tries: int = 1, dpi: int = 150):
    """Worker task: Render diagram from DSL (CPU-intensive)"""
    logger.info(f"[Worker] Rendering diagram for task {task_id}")

    try:
        from src.services.diagram.rendering import render_dsl_to_image

        img_b64 = render_dsl_to_image(dsl, epochs=epochs, n_tries=n_tries, dpi=dpi)
        logger.info(f"[Worker {task_id}] Completed")

        return {
            "task_id": task_id,
            "status": "completed",
            "image": img_b64,
        }

    except Exception as e:
        logger.error(f"[Worker {task_id}] Failed: {e}")
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e),
        }

