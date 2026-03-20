from abc import ABC, abstractmethod

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

    def __str__(self) -> str:
        if isinstance(self.val, str):
            return self.val
        x, rest = self.val
        return f"({x} {' '.join(map(str, rest))})"


class Line(Primitive):

    def points_on(self):
        if isinstance(self.val, str):
            return []
        head, args = self.val
        if head == "through":
            return [args[0]]
        return []

    def __str__(self) -> str:
        if isinstance(self.val, str):
            return self.val
        head, args = self.val
        if head == "through":
            return f"(through {args[0]})"
        return f"({head} {' '.join(map(str, args))})"


class Triangle(Primitive):

    def __init__(self, points: list[Point]):
        if len(points) != 3:
            raise ValueError("Triangle must have exactly 3 points")
        self.points = points
        super().__init__(tuple(p.val for p in points))

    def __str__(self) -> str:
        return f"Triangle({', '.join(map(str, self.points))})"


class Circle(Primitive):

    def __str__(self) -> str:
        if isinstance(self.val, str):
            return self.val
        head, args = self.val
        return f"({head} {' '.join(map(str, args))})"
