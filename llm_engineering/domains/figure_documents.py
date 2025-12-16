"""Domain models for figure extraction."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np

from llm_engineering.domains.orm.nosql import NoSQLBaseDocument


@dataclass
class BBox:
    """Bounding box [x1, y1, x2, y2]."""
    coords: List[int]

    @property
    def x1(self) -> int:
        return self.coords[0]

    @property
    def y1(self) -> int:
        return self.coords[1]

    @property
    def x2(self) -> int:
        return self.coords[2]

    @property
    def y2(self) -> int:
        return self.coords[3]

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def crop(self, image: np.ndarray) -> np.ndarray:
        """Crop image using bbox."""
        return image[self.y1:self.y2, self.x1:self.x2]


@dataclass
class Caption:
    """Figure caption."""
    bbox: BBox
    text: str = ""


@dataclass
class Figure:
    """Detected figure."""
    bbox: BBox
    caption: Optional[Caption] = None
    page: int = 0


@dataclass
class ExtractedFigure:
    """Extracted figure result."""
    path: Path
    caption: str
    bbox: List[int]

    @classmethod
    def from_figure(cls, figure: Figure, save_path: Path) -> "ExtractedFigure":
        """Create from Figure object."""
        return cls(
            path=save_path,
            caption=figure.caption.text if figure.caption else "",
            bbox=figure.bbox.coords
        )

    def to_dict(self) -> dict:
        """Export to dict."""
        return {
            "image_path": str(self.path),
            "caption": self.caption
        }


class FigureDocument(NoSQLBaseDocument):
    source_pdf: str
    figure_path: str
    caption: str
    bbox: List[int]
    page: int

    class Settings:
        name = "figures"

