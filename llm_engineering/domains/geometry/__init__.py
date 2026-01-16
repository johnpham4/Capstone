"""
Domain geometry package
"""
from llm_engineering.domains.geometry.entities import GeometricPoint, Diagram
from llm_engineering.domains.geometry.value_objects import Point, Line, Triangle, Circle, Primitive
from llm_engineering.domains.geometry.types import (
    DiagramType,
    TriangleType,
    QuadrilateralType,
    CircleType,
    ConstraintType
)

__all__ = [
    # Entities
    'GeometricPoint',
    'Diagram',

    # Value Objects
    'Point',
    'Line',
    'Triangle',
    'Circle',
    'Primitive',

    # Types
    'DiagramType',
    'TriangleType',
    'QuadrilateralType',
    'CircleType',
    'ConstraintType',
]
