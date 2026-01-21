import math
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class GeometricPoint:
    x: float
    y: float
    name: Optional[str] = None

    def distance_to(self, other: 'GeometricPoint') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def __str__(self) -> str:
        if self.name:
            return f"{self.name}({self.x:.2f}, {self.y:.2f})"
        return f"({self.x:.2f}, {self.y:.2f})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, GeometricPoint):
            return False
        return (self.x == other.x and self.y == other.y and
                self.name == other.name)

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.name))


@dataclass
class Diagram:
    """
    Entity/Aggregate Root: Represents a complete geometric diagram
    Manages all geometric objects and their relationships
    """
    points: Dict[str, GeometricPoint] = field(default_factory=dict)
    triangles: List[tuple] = field(default_factory=list)
    quadrilaterals: List[List[GeometricPoint]] = field(default_factory=list)
    segments: List[tuple] = field(default_factory=list)
    circles: List[tuple] = field(default_factory=list)  # (center, radius_or_info)
    lines: Dict[str, Any] = field(default_factory=dict)
    angle_bisectors: List[Dict] = field(default_factory=list)
    angle_equal_assertions: List[Dict] = field(default_factory=list)  # Store angle-equal constraints

    # Tick marks for equal segments
    tick_styles: List[str] = field(default_factory=lambda: ["k-", "k--", "kx", "kxx", "kg", "k---"])
    segment_group: Dict[frozenset, int] = field(default_factory=dict)
    group_tick: Dict[int, str] = field(default_factory=dict)

    def add_point(self, name: str, point: GeometricPoint) -> None:
        """Add a named point to the diagram"""
        self.points[name] = point

    def add_triangle(self, p1: GeometricPoint, p2: GeometricPoint, p3: GeometricPoint,
                    equal_sides: Optional[List[tuple]] = None, 
                    right_angle_at: Optional[int] = None,
                    equal_angles: Optional[List[tuple]] = None) -> None:
        """
        Add a triangle to the diagram
        equal_sides: list of tuples indicating which sides are equal, e.g. [(0,1), (0,2)]
        right_angle_at: vertex index with right angle (0, 1, or 2)
        equal_angles: list of tuples indicating which angles are equal, e.g. [(0, 1)] means angle at vertex 0 = angle at vertex 1
        """
        self.triangles.append((p1, p2, p3, equal_sides, right_angle_at, equal_angles))

    def add_quadrilateral(self, p1, p2, p3, p4, metadata):
        """Add a quadrilateral to the diagram"""
        quadrilateral = {
            'points': [p1, p2, p3, p4],
            'type': metadata.get('type', 'general')
        }
        if not hasattr(self, 'quadrilaterals'):
            self.quadrilaterals = []
        self.quadrilaterals.append(quadrilateral)
        

    def add_circle(self, center: GeometricPoint, info: Any) -> None:
        """Add a circle to the diagram"""
        self.circles.append((center, info))

    def add_segment(self, p1: GeometricPoint, p2: GeometricPoint, color: str = "black") -> None:
        """Add a line segment between two points"""
        self.segments.append((p1, p2, color))

    def add_line(self, name: str, line: Any) -> None:
        """Add a named line to the diagram"""
        self.lines[name] = line

    def mark_equal_segments(self, segments: List[tuple]) -> None:
        """
        Mark multiple segments as equal by assigning them a tick style
        segments: list of (p1, p2) tuples
        """
        group_id = len(self.group_tick)

        if group_id >= len(self.tick_styles):
            raise ValueError("No tick style available")

        self.group_tick[group_id] = self.tick_styles[group_id]

        for p1, p2 in segments:
            key = frozenset({p1, p2})
            self.segment_group[key] = group_id

    def get_tick_style(self, p1: GeometricPoint, p2: GeometricPoint) -> Optional[str]:
        """Get the tick style for a segment if it's marked as equal to others"""
        key = frozenset({p1, p2})
        if key not in self.segment_group:
            return None
        group_id = self.segment_group[key]
        return self.group_tick[group_id]

    def to_dict(self) -> Dict[str, Any]:
        """Export diagram data as dictionary"""
        return {
            "points": {
                name: {"x": pt.x, "y": pt.y, "name": pt.name}
                for name, pt in self.points.items()
            },
            "triangles": len(self.triangles),
            "quadrilaterals": len(self.quadrilaterals),
            "segments": len(self.segments),
            "lines": len(self.lines)
        }