"""Diagram rendering — pure business logic (no Celery dependency).

Called by:
    - Celery worker task (infrastructures/celery/tasks.py)
    - DiagramService.render_blocking() for sync path
"""

import io
import base64

import matplotlib.pyplot as plt
from loguru import logger

from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.optimizer_old import Optimizer
from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer


def render_dsl_to_image(
    dsl: str,
    epochs: int = 500,
    n_tries: int = 1,
    dpi: int = 150,
) -> str:
    """Render DSL string to base64-encoded PNG image.

    Raises:
        RuntimeError: on any rendering failure.
    Returns:
        base64-encoded PNG string.
    """
    dsl_lines = dsl.split("\n") if "\n" in dsl else [dsl]
    dsl_lines = [line.strip() for line in dsl_lines if line.strip()]

    builder = DiagramBuilder(dsl_lines)
    opts = {
        "epochs": epochs,
        "n_tries": n_tries,
        "learning_rate": 0.01,
        "eps": 1e-6,
        "seed": 42,
    }
    optimizer = Optimizer(builder.instructions, opts, verbosity=False)
    diagram = optimizer.solve()

    renderer = MatplotlibDiagramRenderer()
    fig, ax = renderer.render(diagram, save=False, show=False)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    logger.debug(f"Rendered diagram: {len(buf.getvalue())} bytes")
    return img_b64
