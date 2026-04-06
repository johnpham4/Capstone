import base64
import os
import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import boto3
from loguru import logger

from src.config.settings.base import settings
from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer
from src.services.diagram.optimizer import Optimizer


class DiagramService:
    def __init__(self):
        self.s3_bucket = os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET_NAME")
        self.s3_prefix = os.getenv("S3_DIAGRAM_PREFIX", "diagrams")

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )

    def generate_and_render(
        self,
        task_id: str,
        dsl: str,
        epochs: int = 500,
        n_tries: int = 1,
        dpi: int = 150,
        timeout: int = 60,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        _ = timeout

        try:
            dsl_ms = int((time.perf_counter() - start) * 1000)

            render_start = time.perf_counter()
            image_bytes = self._render_dsl_to_image(
                dsl=dsl,
                task_id=task_id,
                epochs=epochs,
                n_tries=n_tries,
                dpi=dpi,
            )
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            render_ms = int((time.perf_counter() - render_start) * 1000)

            image_name = f"{task_id}_{uuid4().hex}.png"
            s3_url = self._store_image_on_s3(image_bytes=image_bytes, image_name=image_name)

            return {
                "status": "success",
                "dsl": dsl,
                "image_base64": image_base64,
                "image": image_base64,
                "s3_url": s3_url,
                "generation_time_ms": dsl_ms,
                "render_time_ms": render_ms,
            }
        except Exception as e:
            logger.exception(f"Diagram generation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
            }

    def _render_dsl_to_image(
        self,
        dsl: str,
        task_id: str,
        epochs: int,
        n_tries: int,
        dpi: int,
    ) -> bytes:
        lines = self._normalize_dsl_lines(dsl)
        diagram = self._build_diagram(lines=lines, epochs=epochs, n_tries=n_tries)
        image_path = self._render_diagram_file(diagram=diagram, task_id=task_id, dpi=dpi)

        image_bytes = image_path.read_bytes()
        if not image_bytes:
            raise RuntimeError("Rendered image is empty")

        return image_bytes

    @staticmethod
    def _normalize_dsl_lines(dsl: str) -> list[str]:
        lines = [line.strip() for line in dsl.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError("DSL is empty after normalization")
        return lines

    @staticmethod
    def _build_diagram(lines: list[str], epochs: int, n_tries: int):
        builder = DiagramBuilder(lines)
        optimizer_opts = {
            "epochs": epochs,
            "n_tries": n_tries,
            "learning_rate": settings.DIAGRAM_OPTIMIZER_LR,
            "seed": 42,
        }
        optimizer = Optimizer(builder.instructions, optimizer_opts, verbosity=False)
        return optimizer.solve(n_tries=n_tries)

    @staticmethod
    def _render_diagram_file(diagram, task_id: str, dpi: int) -> Path:
        output_dir = Path(settings.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"{task_id}.png"

        renderer = MatplotlibDiagramRenderer(diagram)
        renderer.render(diagram=diagram, show=False, save=True, filename=str(image_path))
        return image_path

    def _store_image_on_s3(self, image_bytes: bytes, image_name: str) -> Optional[str]:
        if not self.s3_bucket:
            logger.warning("S3 bucket is not configured (set AWS_S3_BUCKET or S3_BUCKET_NAME)")
            return None

        s3_key = f"{self.s3_prefix.rstrip('/')}/{image_name}"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=image_bytes,
            ContentType="image/png",
        )

        return f"s3://{self.s3_bucket}/{s3_key}"

    def generation(
        self,
        dsl: str,
        task_id: Optional[str] = None,
        epochs: int = 500,
        n_tries: int = 1,
        dpi: int = 150,
    ) -> dict[str, Any]:
        return self.generate_and_render(
            task_id=task_id or f"generation_{uuid4().hex[:8]}",
            dsl=dsl,
            epochs=epochs,
            n_tries=n_tries,
            dpi=dpi,
        )