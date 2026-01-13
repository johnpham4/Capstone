from abc import ABC, abstractmethod
from typing import Iterable
import math

class GeometricPoint:
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self.name = name

    def distance_to(self, other):
        """Calculate distance to another point"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __str__(self):
        if self.name:
            return f"{self.name}({self.x:.2f}, {self.y:.2f})"
        return f"({self.x:.2f}, {self.y:.2f})"


class Primitive(ABC):
    def __init__(self, val):
        self.val = val

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Primitive):
            return NotImplemented
        return type(self) is type(other) and self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

    @abstractmethod
    def __str__(self) -> str:
        ...


class Point(Primitive):
    """Represents a geometric point"""

    def __str__(self) -> str:
        if isinstance(self.val, str):
            return self.val
        x, rest = self.val
        return f"({x} {' '.join(map(str, rest))})"


class Line(Primitive):
    def points_on(self):
        if isinstance(self.val, str):
            return []

        pred, points = self.val

        if pred == "connecting":
            return points
        elif pred in ("paraAt", "perpAt"):
            return [points[0]]
        elif pred == "mediator":
            return []
        return []

    def __str__(self) -> str:
        if isinstance(self.val, str):
            return self.val

        pred, args = self.val
        return f"({pred} {' '.join(map(str, args))})"


class Triangle(Primitive):
    def __init__(self, points: list[Point]):
        if len(points) != 3:
            raise ValueError("Triangle must have exactly 3 points")

        self.points = points
        super().__init__(tuple(p.val for p in points))

    def __str__(self) -> str:
        return f"Triangle({', '.join(map(str, self.points))})"
