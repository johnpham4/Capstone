import time
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger

from src.config.settings import settings
from src.services.diagram.diagram_builder import DiagramBuilder
from src.services.diagram.matplotlib_renderer import MatplotlibDiagramRenderer
from src.services.diagram.optimizer import Optimizer


class DiagramService:
    def __init__(self):
        self.s3_bucket = settings.S3_BUCKET_NAME
        self.s3_prefix = settings.S3_DIAGRAM_PREFIX
        self.aws_region = settings.AWS_REGION or settings.REGION_NAME or "us-east-1"

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=self.aws_region,
        )

    def generate_and_render(
        self,
        task_id: str,
        dsl: str,
        epochs: int = 3000,
        n_tries: int = 3,
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
            render_ms = int((time.perf_counter() - render_start) * 1000)

            image_name = f"{task_id}_{uuid4().hex}.png"
            s3_url = self._store_image_on_s3(image_bytes=image_bytes, image_name=image_name)
            image_url = self._build_image_access_url(s3_url)

            return {
                "status": "success",
                "dsl": dsl,
                "image_url": image_url,
                "s3_url": s3_url,
                "generation_time_ms": dsl_ms,
                "render_time_ms": render_ms,
            }
        except Exception as e:
            logger.exception(f"Diagram generation failed: {e}")
            return {
                "status": "failed",
                "error_code": self._classify_error_code(e),
                "error": str(e),
            }

    @staticmethod
    def _classify_error_code(error: Exception) -> str:
        message = str(error).lower()

        if isinstance(error, (ClientError, BotoCoreError)) or "s3" in message or "bucket" in message:
            return "STORAGE_ERROR"

        if "dsl" in message or "s-expression" in message or "parse" in message:
            return "DSL_PARSE_ERROR"

        if "render" in message or "image" in message or "matplotlib" in message:
            return "RENDER_ERROR"

        return "DIAGRAM_GENERATION_ERROR"

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
            raise RuntimeError("S3 bucket is not configured (set AWS_S3_BUCKET or S3_BUCKET_NAME)")

        s3_key = f"{self.s3_prefix.rstrip('/')}/{image_name}"
        self.s3_client.put_object(
            Bucket=self.s3_bucket,
            Key=s3_key,
            Body=image_bytes,
            ContentType="image/png",
        )

        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        logger.info(f"Uploaded diagram image to {s3_uri}")

        return s3_uri

    def _build_image_access_url(self, s3_uri: Optional[str]) -> Optional[str]:
        if not s3_uri or not s3_uri.startswith("s3://"):
            return s3_uri

        path = s3_uri.removeprefix("s3://")
        if "/" not in path:
            return None

        bucket, key = path.split("/", 1)

        # Private buckets need a signed URL for browser rendering.
        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )

    def generation(
        self,
        dsl: str,
        task_id: Optional[str] = None,
        epochs: int = 3000,
        n_tries: int = 3,
        dpi: int = 150,
    ) -> dict[str, Any]:
        return self.generate_and_render(
            task_id=task_id or f"generation_{uuid4().hex[:8]}",
            dsl=dsl,
            epochs=epochs,
            n_tries=n_tries,
            dpi=dpi,
        )