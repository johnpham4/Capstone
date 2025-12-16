"""Figure detection using DocLayout-YOLO."""

from pathlib import Path
from typing import List
import numpy as np
import torch
from doclayout_yolo import YOLOv10
from huggingface_hub import snapshot_download
from loguru import logger

from llm_engineering.applications.networks.base import SingletonMeta
from llm_engineering.domains.figure_documents import BBox, Caption, Figure


class FigureDetector(metaclass=SingletonMeta):
    def __init__(self, model_path: str, device: str = "auto", model_id: str = None):
        """
        Initialize figure detector.

        Args:
            model_path: Local path to model directory
            device: Compute device (auto/cuda/cpu)
            model_id: HuggingFace model ID for downloading if not exists
        """
        model_path = Path(model_path)

        # Download model if not exists
        if not model_path.exists() and model_id:
            logger.info(f"Downloading model from {model_id} to {model_path}")
            snapshot_download(model_id, local_dir=str(model_path))
            logger.info("Model downloaded successfully")

        # Find the .pt file in the directory
        pt_files = list(model_path.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt model file found in {model_path}")

        model_file = pt_files[0]
        logger.info(f"Loading model from {model_file}")

        self.model = YOLOv10(str(model_file))
        self.device = self._get_device(device)
        self.model.to(self.device)

    def _get_device(self, device: str) -> str:
        """Get compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def __call__(
        self,
        image_path: str | Path,
        conf: float = 0.25,
        imgsz: int = 1280
    ) -> tuple[List[Figure], List[Caption]]:
        """Detect figures and captions in image.

        Returns:
            (figures, captions) - separate lists for matching later
        """
        results = self.model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            device=self.device,
            verbose=False
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        names = self.model.names

        figures = []
        captions = []

        for box, cls in zip(boxes, classes):
            class_name = names[int(cls)]
            coords = [int(x) for x in box]
            bbox = BBox(coords)

            if class_name == "figure":
                figures.append(Figure(bbox=bbox))
            elif class_name == "figure_caption":
                captions.append(Caption(bbox=bbox))

        return figures, captions
