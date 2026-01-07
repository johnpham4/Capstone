from abc import ABC, abstractmethod
import numbers
from typing import Union
from .utils import FuncInfo

class Primitive(ABC):
    def __init__(self, val: Union[str, list]):
        self.val = val
        super().__init__()

    def __eq__(self, object: object):
        return type(self) == type(object)

    def __hash__(self):
        return hash(self)


    @abstractmethod
    def __str__(self): ...


class Point(Primitive):
    def __str__(self):
        if isinstance(self.val, str):
            return self.val
        return f"({self.val[0]} {' '.join([str(v) for v in self.val[1]])})"


class Num(Primitive):
    def __str__(self):
        if isinstance(self.val, numbers.Number):
            return str(self.val)
        return f"({self.val[0]} {' '.join([str(v) for v in self.val[1]])})"


class Line(Primitive):
    def pointsOn(self):
        """Get points that define or lie on this line"""
        if isinstance(self.val, FuncInfo):
            pred, points = self.val
            if pred == "connecting":
                return points
            elif pred == "para-at":
                return [points[0]]
            elif pred == "perp-at":
                return [points[0]]
            elif pred == "mediator":
                return list()
            elif pred == "i-bisector":
                return [points[1]]
            else:
                return list()
        return list()

    def __str__(self):
        if isinstance(self.val, str):
            return self.val
        elif isinstance(self.val, FuncInfo):
            pred, args = self.val
            return f"({pred} {' '.join([str(a) for a in args])})"
        else:
            return str(self.val)
