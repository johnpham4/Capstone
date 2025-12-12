from ultralytics import YOLO
from llm_engineering.applications.networks.base import SingletonMeta


class YOLOLayoutDetector(metaclass=SingletonMeta):
    """YOLO model for document layout detection - Singleton"""

    def __init__(self, model_path: str = "yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"):
        self.model = YOLO(model_path)

    def detect(self, image_path: str):
        """Detect layout blocks in image"""
        results = self.model(source=image_path, save=False, verbose=False)
        return results
