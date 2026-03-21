<<<<<<< HEAD
from loguru import logger
from src.infrastructures.celery.config import celery_app


@celery_app.task(bind=True, name="render_diagram")
def render_diagram_task(self, task_id: str, dsl: str, epochs: int = 500, n_tries: int = 1, dpi: int = 150):
    """Worker task: Render diagram from DSL (CPU-intensive)"""
    logger.info(f"[Worker] Rendering diagram for task {task_id}")

    try:
        from src.services.diagram.diagram_builder import DiagramBuilder
        from src.services.diagram.optimizer import Optimizer
        from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer
        import matplotlib.pyplot as plt
        import io
        import base64

        dsl_lines = dsl.split('\n') if '\n' in dsl else [dsl]
        dsl_lines = [line.strip() for line in dsl_lines if line.strip()]

        builder = DiagramBuilder(dsl_lines)
        optimizer = Optimizer(builder.instructions, epochs=epochs, n_tries=n_tries, verbosity=False)
        diagram = optimizer.solve()

        renderer = MatplotlibDiagramRenderer()
        fig, ax = renderer.render(diagram, save=False, show=False)

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        logger.info(f"[Worker {task_id}] Completed: {len(buf.getvalue())} bytes")

        return {
            "task_id": task_id,
            "status": "completed",
            "image": img_b64,
            "size": len(buf.getvalue())
        }

    except Exception as e:
        logger.error(f"[Worker {task_id}] Failed: {e}")
        return {
            "task_id": task_id,
            "status": "failed",
            "error": str(e)
        }

=======
from loguru import logger
from src.infrastructures.celery.config import celery_app


@celery_app.task(bind=True, name="render_diagram", store_errors_even_if_ignored=True)
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

>>>>>>> minh-re
