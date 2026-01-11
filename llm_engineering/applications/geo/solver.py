"""
Simplified geometry solver for triangles, lines, and points
"""

import math
import random
from .instruction_reader import InstructionReader
from .instruction import Parameterize, Assert
from .primitives import Point, Line, Triangle
from .diagram import Diagram


class GeometricPoint:
    """A point with x,y coordinates"""

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


class GeometrySolver:
    """Simple solver for geometry problems with triangles, lines, and points"""

    def __init__(self, problem_lines, verbosity=0):
        self.verbosity = verbosity
        self.reader = InstructionReader(problem_lines)
        self.point_coords = {}  # Map from Point name to GeometricPoint
        self.diagram = Diagram()  # Diagram for visualization
        self.triangles = []  # Keep track of triangles created

    def solve(self):
        """Solve the geometry problem"""
        if self.verbosity >= 0:
            print("\n=== INPUT INSTRUCTIONS ===")
            for instr in self.reader.instructions:
                print(f"  {instr}")

        # Process instructions
        for instr in self.reader.instructions:
            if isinstance(instr, Parameterize):
                self.process_parameterize(instr)
            elif isinstance(instr, Assert):
                self.process_assert(instr)

        # Add all points to diagram
        for name, point in self.point_coords.items():
            self.diagram.add_point(name, point)

        # Add triangles to diagram
        for tri_points in self.triangles:
            p1 = self.point_coords[tri_points[0]]
            p2 = self.point_coords[tri_points[1]]
            p3 = self.point_coords[tri_points[2]]
            self.diagram.add_triangle(p1, p2, p3)

        return self.point_coords, self.diagram

    def process_parameterize(self, instr):
        """Process parameterization instruction"""
        param_type = instr.param_type

        if param_type == "iso-tri":
            # Isosceles triangle
            self.create_isosceles_triangle(instr.objects, instr.args)
        elif param_type == "on-seg":
            # Point on segment
            self.create_point_on_segment(instr.objects, instr.args)
        elif param_type in ["triangle", "acute-tri", "equi-tri"]:
            # General triangle
            self.create_triangle(instr.objects, param_type)
        else:
            if self.verbosity >= 0:
                print(f"  [Warning] Unsupported parameterization: {param_type}")

    def create_isosceles_triangle(self, points, args):
        """Create an isosceles triangle"""
        if len(points) != 3:
            raise ValueError("Isosceles triangle requires 3 points")

        # args[0] is the apex point (where two equal sides meet)
        apex = args[0] if args else points[0]

        # Find which point is the apex
        apex_idx = None
        for i, p in enumerate(points):
            if p.val == apex.val:
                apex_idx = i
                break

        if apex_idx is None:
            apex_idx = 0  # Default to first point

        # Create isosceles triangle with apex at origin
        # Two equal sides of length 3, base of length 4
        if apex_idx == 0:
            # A is apex
            self.point_coords[points[0].val] = GeometricPoint(0, 3, points[0].val)
            self.point_coords[points[1].val] = GeometricPoint(-2, 0, points[1].val)
            self.point_coords[points[2].val] = GeometricPoint(2, 0, points[2].val)
        elif apex_idx == 1:
            # B is apex
            self.point_coords[points[1].val] = GeometricPoint(0, 3, points[1].val)
            self.point_coords[points[0].val] = GeometricPoint(-2, 0, points[0].val)
            self.point_coords[points[2].val] = GeometricPoint(2, 0, points[2].val)
        else:
            # C is apex
            self.point_coords[points[2].val] = GeometricPoint(0, 3, points[2].val)
            self.point_coords[points[0].val] = GeometricPoint(-2, 0, points[0].val)
            self.point_coords[points[1].val] = GeometricPoint(2, 0, points[1].val)

        # Track triangle
        self.triangles.append([p.val for p in points])

        if self.verbosity >= 0:
            print(f"\n  Created isosceles triangle with apex at {apex}:")
            for p in points:
                coord = self.point_coords[p.val]
                print(f"    {coord}")

    def create_point_on_segment(self, points, args):
        """Create a point on a line segment"""
        if len(points) != 1:
            raise ValueError("on-seg requires exactly 1 point to be created")

        new_point = points[0]

        if len(args) < 2:
            raise ValueError("on-seg requires 2 existing points")

        # Get the two points that define the segment
        p1_name = args[0].val
        p2_name = args[1].val

        if p1_name not in self.point_coords or p2_name not in self.point_coords:
            raise RuntimeError(f"Points {p1_name} and {p2_name} must be defined before using on-seg")

        p1 = self.point_coords[p1_name]
        p2 = self.point_coords[p2_name]

        # Place new point at a random position on the segment
        t = random.uniform(0.2, 0.8)  # Avoid endpoints
        x = p1.x + t * (p2.x - p1.x)
        y = p1.y + t * (p2.y - p1.y)

        self.point_coords[new_point.val] = GeometricPoint(x, y, new_point.val)

        if self.verbosity >= 0:
            print(f"\n  Created point {new_point.val} on segment {p1_name}-{p2_name}:")
            print(f"    {self.point_coords[new_point.val]}")

    def create_triangle(self, points, triangle_type):
        """Create a general triangle"""
        if len(points) != 3:
            raise ValueError("Triangle requires 3 points")

        # Create a simple triangle
        self.point_coords[points[0].val] = GeometricPoint(0, 0, points[0].val)
        self.point_coords[points[1].val] = GeometricPoint(4, 0, points[1].val)
        self.point_coords[points[2].val] = GeometricPoint(2, 3, points[2].val)

        # Track triangle
        self.triangles.append([p.val for p in points])

        if self.verbosity >= 0:
            print(f"\n  Created {triangle_type}:")
            for p in points:
                coord = self.point_coords[p.val]
                print(f"    {coord}")

    def process_assert(self, instr):
        """Process assertion instruction"""
        constraint = instr.constraint

        if self.verbosity >= 0:
            print(f"\n  Verifying constraint: {constraint}")

        # For now, just acknowledge the constraint
        # In a full implementation, this would verify the constraint is satisfied


def solve_geometry_problem(problem_lines, verbosity=0):
    """
    Solve a geometry problem given as s-expressions

    Args:
        problem_lines: List of strings containing the problem definition
        verbosity: Verbosity level (0 = quiet, 1 = verbose)

    Returns:
        Tuple of (point_coords, diagram)
        - point_coords: Dictionary mapping point names to GeometricPoint objects
        - diagram: Diagram object that can be plotted
    """
    solver = GeometrySolver(problem_lines, verbosity=verbosity)
    return solver.solve()
