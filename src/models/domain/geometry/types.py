<<<<<<<< HEAD:src/models/domain/geometry/types.py
from enum import StrEnum, auto


class DiagramType(StrEnum):
    TRIANGLE = auto()
    QUADRILATERAL = auto()
    CIRCLE = auto()
    POLYGON = auto()
    POINT = auto()
    SEGMENT = auto()
    LINE = auto()


class TriangleType(StrEnum):
    SCALENE = auto()        # tam giác thường
    ISOSCELES = auto()      # cân
    EQUILATERAL = auto()    # đều
    RIGHT = auto()          # vuông
    RIGHT_ISOSCELES = auto() # vuông cân


class QuadrilateralType(StrEnum):
    GENERAL = auto()
    TRAPEZOID = auto()       # hình thang
    PARALLELOGRAM = auto()   # hình bình hành
    RECTANGLE = auto()       # hình chữ nhật
    RHOMBUS = auto()         # hình thoi
    SQUARE = auto()          # hình vuông


class CircleType(StrEnum):
    GENERAL = auto()
    CIRCUMCIRCLE = auto()   # đường tròn ngoại tiếp
    INCIRCLE = auto()       # đường tròn nội tiếp
    EXCIRCLE = auto()       # đường tròn bàng tiếp


class ConstraintType(StrEnum):
    EQUAL_LENGTH = auto()    # cong - độ dài bằng nhau
    EQUAL_ANGLE = auto()     # góc bằng nhau
    PERPENDICULAR = auto()   # perp - vuông góc
    PARALLEL = auto()        # para - song song
    COLLINEAR = auto()       # thẳng hàng
    ON_LINE = auto()         # on-line - nằm trên đường
    ON_SEGMENT = auto()      # on-seg - nằm trên đoạn
    ON_CIRCLE = auto()       # on-circ - nằm trên đường tròn
    MIDPOINT = auto()        # midp - trung điểm
========
from enum import StrEnum, auto


class DiagramType(StrEnum):
    TRIANGLE = auto()
    QUADRILATERAL = auto()
    CIRCLE = auto()
    POLYGON = auto()
    POINT = auto()
    SEGMENT = auto()
    LINE = auto()


class TriangleType(StrEnum):
    SCALENE = auto()        # tam giác thường
    ISOSCELES = auto()      # cân
    EQUILATERAL = auto()    # đều
    RIGHT = auto()          # vuông
    RIGHT_ISOSCELES = auto() # vuông cân


class QuadrilateralType(StrEnum):
    GENERAL = auto()
    TRAPEZOID = auto()       # hình thang
    PARALLELOGRAM = auto()   # hình bình hành
    RECTANGLE = auto()       # hình chữ nhật
    RHOMBUS = auto()         # hình thoi
    SQUARE = auto()          # hình vuông


class CircleType(StrEnum):
    GENERAL = auto()
    CIRCUMCIRCLE = auto()   # đường tròn ngoại tiếp
    INCIRCLE = auto()       # đường tròn nội tiếp
    EXCIRCLE = auto()       # đường tròn bàng tiếp


class ConstraintType(StrEnum):
    EQUAL_LENGTH = auto()    # cong - độ dài bằng nhau
    EQUAL_ANGLE = auto()     # góc bằng nhau
    PERPENDICULAR = auto()   # perp - vuông góc
    PARALLEL = auto()        # para - song song
    COLLINEAR = auto()       # thẳng hàng
    ON_LINE = auto()         # on-line - nằm trên đường
    ON_SEGMENT = auto()      # on-seg - nằm trên đoạn
    ON_CIRCLE = auto()       # on-circ - nằm trên đường tròn
    MIDPOINT = auto()        # midp - trung điểm
>>>>>>>> minh-re:src/services/diagram/model/types.py
