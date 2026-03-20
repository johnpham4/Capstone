"""Geometry domain — dataclasses and types cho diagram pipeline.

Đây KHÔNG phải DDD domain. Đây là typed data structures
cho DSL parser → optimizer → renderer pipeline.
"""

from .entities import GeometricPoint, Diagram
from .value_objects import Point, Line, Triangle, Circle, Primitive
from .types import DiagramType, TriangleType, QuadrilateralType, CircleType
from .instructions import Parameter, Assertion, Definition

__all__ = [
    "GeometricPoint",
    "Diagram",
    "Point",
    "Line",
    "Triangle",
    "Circle",
    "Primitive",
    "DiagramType",
    "TriangleType",
    "QuadrilateralType",
    "CircleType",
    "Parameter",
    "Assertion",
    "Definition",
]
