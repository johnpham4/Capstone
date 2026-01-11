"""
Simplified geometry primitives for triangle, line, and point
"""

from abc import ABC, abstractmethod


class Primitive(ABC):
    def __init__(self, val):
        self.val = val
        super().__init__()

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return self.val == other.val

    def __hash__(self):
        return hash(self.val)

    @abstractmethod
    def __str__(self):
        pass


class Point(Primitive):
    """Represents a geometric point"""

    def __str__(self):
        if isinstance(self.val, str):
            return self.val
        else:
            return f"({self.val[0]} {' '.join([str(v) for v in self.val[1]])})"


class Line(Primitive):
    """Represents a geometric line"""

    def points_on(self):
        """Returns points that lie on this line"""
        if isinstance(self.val, str):
            return []
        pred, points = self.val
        if pred == "connecting":
            return points
        elif pred in ["paraAt", "perpAt"]:
            return [points[0]]
        elif pred == "mediator":
            return []
        else:
            return []

    def __str__(self):
        if isinstance(self.val, str):
            return self.val
        else:
            pred, args = self.val
            return f"({pred} {' '.join([str(a) for a in args])})"


class Triangle(Primitive):
    """Represents a geometric triangle"""

    def __init__(self, points):
        """Initialize triangle with 3 points"""
        if len(points) != 3:
            raise ValueError("Triangle must have exactly 3 points")
        self.points = points
        super().__init__(tuple(p.val if isinstance(p, Point) else p for p in points))

    def __str__(self):
        return f"Triangle({', '.join([str(p) for p in self.points])})"
