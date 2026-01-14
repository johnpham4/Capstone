from enum import StrEnum, auto

class TriangleType(StrEnum):
    SCALENE = auto()        # tam giác thường
    ISOSCELES = auto()     # cân
    EQUILATERAL = auto()   # đều
    RIGHT = auto()         # vuông
    RIGHT_ISOSCELES = auto()


class QuadrilateralType(StrEnum):
    GENERAL = auto()
    TRAPEZOID = auto()
    PARALLELOGRAM = auto()
    RECTANGLE = auto()
    RHOMBUS = auto()
    SQUARE = auto()


class CircleType(StrEnum):
    GENERAL = auto()
    CIRCUMCIRCLE = auto()   # ngoại tiếp
    INCIRCLE = auto()       # nội tiếp
