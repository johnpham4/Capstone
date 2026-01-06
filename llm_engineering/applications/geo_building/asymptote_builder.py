"""Asymptote-style geometry builder using PyTorch optimization"""
from typing import List, Dict, Tuple
import numpy as np
import math

from asymptote_parser import AsymptoteParser
from pytorch_optimizer import GeometryOptimizer
from diagram import Diagram
from primitives import Point, Line
from util import FuncInfo
from instruction import Sample, Compute, Parameterize, Assert


class SimplePoint:
    """Simple 2D point"""
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)

    def __add__(self, other):
        return SimplePoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return SimplePoint(self.x - other.x, self.y - other.y)

    def smul(self, scalar):
        return SimplePoint(self.x * scalar, self.y * scalar)


class AsymptoteBuilder:
    """Builder for Asymptote-style geometry with PyTorch optimization"""

    def __init__(self, lines: List[str], optimize=True, n_iterations=1000, lr=0.01):
        self.parser = AsymptoteParser(lines)
        self.optimize = optimize
        self.n_iterations = n_iterations
        self.lr = lr

        self.named_points: Dict[Point, SimplePoint] = {}
        self.named_lines: Dict[Line, Tuple] = {}
        self.unnamed_points: List[SimplePoint] = []
        self.unnamed_lines: List[Tuple] = []
        self.constraints = []

    def build(self) -> Diagram:
        # Track which points are computed (not free)
        self.computed_points = set()
        # Track on-segment constraints for parameterized points
        self.on_segment_constraints = {}  # {point: (p1, p2)}

        for instr in self.parser.instructions:
            if isinstance(instr, Sample):
                self.sample_triangle(instr.points, instr.sampler, instr.args)
            elif isinstance(instr, Parameterize):
                self.parameterize_point(instr.obj_name, instr.parameterization)
            elif isinstance(instr, Compute):
                if isinstance(instr.obj_name, Point):
                    self.computed_points.add(instr.obj_name)
                    self.compute_point(instr.obj_name, instr.computation)
                elif isinstance(instr.obj_name, Line):
                    self.compute_line(instr.obj_name, instr.computation)
            elif isinstance(instr, Assert):
                self.constraints.append(instr.constraint)

        # Optimize if enabled
        if self.optimize and self.constraints:
            # Only optimize if there are free points (non-computed)
            free_points = [p for p in self.named_points.keys() if p not in self.computed_points]
            if free_points:
                self.optimize_with_pytorch()
            else:
                print("  (No free points to optimize - all points are computed)")

        # Convert segments
        segments = []
        for p1, p2 in self.parser.segments:
            sp1 = self.named_points.get(p1)
            sp2 = self.named_points.get(p2)
            if sp1 and sp2:
                segments.append((sp1, sp2))

        return Diagram(
            named_points=self.named_points,
            named_lines=self.named_lines,
            segments=segments,
            seg_colors=self.parser.seg_colors,
            unnamed_points=self.unnamed_points,
            unnamed_lines=self.unnamed_lines,
            ndgs={},
            goals={}
        )

    def sample_triangle(self, points: List[Point], sampler: str, args: Tuple):
        """Initialize triangle with random coordinates"""
        if sampler == "triangle":
            # Regular triangle: random initialization
            for p in points:
                x = np.random.uniform(-2, 2)
                y = np.random.uniform(-1, 3)
                self.named_points[p] = SimplePoint(x, y)

        elif sampler == "right-tri":
            # Right triangle at special vertex
            special_p = args[0] if args else points[0]
            special_idx = points.index(special_p)

            # Right angle vertex at origin
            self.named_points[points[special_idx]] = SimplePoint(0, 0)

            # Other two vertices on perpendicular axes
            remaining = [p for i, p in enumerate(points) if i != special_idx]
            self.named_points[remaining[0]] = SimplePoint(np.random.uniform(1, 3), 0)
            self.named_points[remaining[1]] = SimplePoint(0, np.random.uniform(1, 3))

        elif sampler == "equi-tri":
            # Equilateral triangle
            base_y = np.random.uniform(-1, 0)
            base_left = np.random.uniform(-2, -1)
            base_right = np.random.uniform(1, 2)
            base_len = base_right - base_left

            self.named_points[points[0]] = SimplePoint(base_left, base_y)
            self.named_points[points[1]] = SimplePoint(base_right, base_y)
            self.named_points[points[2]] = SimplePoint(
                (base_left + base_right) / 2,
                base_y + base_len * math.sqrt(3) / 2
            )

        elif sampler == "iso-tri":
            # Isosceles triangle with special vertex
            special_p = args[0] if args else points[0]
            special_idx = points.index(special_p)

            # Base vertices
            remaining = [p for i, p in enumerate(points) if i != special_idx]
            base_y = np.random.uniform(-1, 0)
            self.named_points[remaining[0]] = SimplePoint(np.random.uniform(-2, -1), base_y)
            self.named_points[remaining[1]] = SimplePoint(np.random.uniform(1, 2), base_y)

            # Special vertex on perpendicular bisector
            base_mid_x = (self.named_points[remaining[0]].x + self.named_points[remaining[1]].x) / 2
            self.named_points[points[special_idx]] = SimplePoint(
                base_mid_x,
                base_y + np.random.uniform(2, 3)
            )

        else:
            # Default: random triangle
            for p in points:
                x = np.random.uniform(-2, 2)
                y = np.random.uniform(-1, 3)
                self.named_points[p] = SimplePoint(x, y)

    def parameterize_point(self, point: Point, param):
        """Create free point with parameterization constraint"""
        param_type = param[0]

        if param_type == 'on-seg':
            # Point on segment between p1 and p2
            p1, p2 = param[1], param[2]
            sp1 = self.named_points.get(p1)
            sp2 = self.named_points.get(p2)

            # Store constraint
            self.on_segment_constraints[point] = (p1, p2)

            if sp1 and sp2:
                # Initialize at random position on segment
                t = np.random.uniform(0.2, 0.8)
                x = sp1.x * (1 - t) + sp2.x * t
                y = sp1.y * (1 - t) + sp2.y * t
                self.named_points[point] = SimplePoint(x, y)
            else:
                # Fallback
                self.named_points[point] = SimplePoint(
                    np.random.uniform(-1, 1),
                    np.random.uniform(-1, 1)
                )

        elif param_type == 'on-line':
            # Point on line
            line = param[1]
            # Initialize randomly, will be constrained later
            self.named_points[point] = SimplePoint(
                np.random.uniform(-2, 2),
                np.random.uniform(-1, 3)
            )

        else:
            # Default: random point
            self.named_points[point] = SimplePoint(
                np.random.uniform(-2, 2),
                np.random.uniform(-1, 3)
            )

    def compute_point(self, point: Point, computation: FuncInfo):
        """Compute point from function"""
        if not isinstance(computation, FuncInfo):
            self.named_points[point] = SimplePoint(
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            )
            return

        func_name = computation.head
        args = computation.args

        if func_name == "midpoint":
            p1, p2 = args[0], args[1]
            sp1 = self.named_points.get(p1)
            sp2 = self.named_points.get(p2)

            if sp1 and sp2:
                self.named_points[point] = SimplePoint(
                    (sp1.x + sp2.x) / 2,
                    (sp1.y + sp2.y) / 2
                )

        elif func_name == "orthocenter":
            # For orthocenter, initialize randomly and let optimizer adjust
            self.named_points[point] = SimplePoint(
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            )

        elif func_name == "centroid":
            p1, p2, p3 = args[0], args[1], args[2]
            sp1 = self.named_points.get(p1)
            sp2 = self.named_points.get(p2)
            sp3 = self.named_points.get(p3)

            if sp1 and sp2 and sp3:
                self.named_points[point] = SimplePoint(
                    (sp1.x + sp2.x + sp3.x) / 3,
                    (sp1.y + sp2.y + sp3.y) / 3
                )

        else:
            # Unknown computation: random initialization
            self.named_points[point] = SimplePoint(
                np.random.uniform(-1, 1),
                np.random.uniform(-1, 1)
            )

    def compute_line(self, line: Line, computation: FuncInfo):
        """Compute line from function"""
        if not isinstance(computation, FuncInfo):
            return

        func_name = computation.head
        args = computation.args

        if func_name == "line":
            # Line through two points
            p1, p2 = args[0], args[1]
            sp1 = self.named_points.get(p1)
            sp2 = self.named_points.get(p2)

            if sp1 and sp2:
                # Store as (normal, distance) form: nx*x + ny*y = r
                dx = sp2.x - sp1.x
                dy = sp2.y - sp1.y
                length = math.sqrt(dx**2 + dy**2) + 1e-8

                # Normal vector (perpendicular to direction)
                nx = -dy / length
                ny = dx / length

                # Distance from origin
                r = nx * sp1.x + ny * sp1.y

                self.named_lines[line] = ((nx, ny), r)

        elif func_name == "perp":
            # Perpendicular line through point
            through_point = args[0]
            ref_line = args[1]

            sp = self.named_points.get(through_point)
            ref = self.named_lines.get(ref_line)

            if sp and ref:
                (nx, ny), r = ref
                # Perpendicular: swap and negate normal
                perp_nx = ny
                perp_ny = -nx
                perp_r = perp_nx * sp.x + perp_ny * sp.y

                self.named_lines[line] = ((perp_nx, perp_ny), perp_r)

    def optimize_with_pytorch(self):
        """Optimize point positions using PyTorch"""
        # Collect all points that need optimization (exclude computed points)
        opt_points = [p for p in self.named_points.keys() if p not in self.computed_points]

        if not opt_points:
            return

        # Create point index map
        point_map = {p: i for i, p in enumerate(opt_points)}

        # Initial coordinates
        initial_coords = {p: (self.named_points[p].x, self.named_points[p].y)
                         for p in opt_points}

        # Create optimizer
        optimizer = GeometryOptimizer(opt_points, initial_coords)

        # Run optimization
        optimized_coords = optimizer.optimize(
            self.constraints,
            point_map,
            line_coords={},
            on_segment_constraints=self.on_segment_constraints,
            n_iterations=self.n_iterations,
            lr=self.lr
        )

        # Update points with optimized coordinates
        for p, (x, y) in optimized_coords.items():
            self.named_points[p] = SimplePoint(x, y)

        # Recompute computed points after optimization
        for instr in self.parser.instructions:
            if isinstance(instr, Compute) and isinstance(instr.obj_name, Point):
                if instr.obj_name in self.computed_points:
                    self.compute_point(instr.obj_name, instr.computation)

        # Update lines that depend on optimized points
        for line, computation in [(instr.obj_name, instr.computation)
                                  for instr in self.parser.instructions
                                  if isinstance(instr, Compute) and isinstance(instr.obj_name, Line)]:
            self.compute_line(line, computation)
