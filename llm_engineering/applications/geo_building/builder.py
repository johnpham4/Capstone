"""Simple geometry builder with optimization - triangles and lines only"""
from typing import List, Dict, Any, Tuple
from instruction_reader import InstructionReader
from diagram import Diagram
from primitives import Point, Line
from util import FuncInfo
import math
import numpy as np
from scipy.optimize import minimize


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


class SimpleBuilder:
    """Simple builder for geometric diagrams with optimization - triangles and lines only"""

    def __init__(self, lines: List[str], optimize=True, n_iterations=100):
        self.reader = InstructionReader(lines)
        self.named_points: Dict[Point, SimplePoint] = {}
        self.named_lines: Dict[Line, tuple] = {}
        self.unnamed_points: List[SimplePoint] = []
        self.unnamed_lines: List[tuple] = []
        self.optimize = optimize
        self.n_iterations = n_iterations
        self.constraints = []  # Store constraints to optimize
        self.point_vars = {}  # Map points to variable indices

    def build(self) -> Diagram:
        """Build the diagram from instructions"""
        # First pass: collect all parameterized points
        for instr in self.reader.instructions:
            if hasattr(instr, 'points') and hasattr(instr, 'sampler'):
                self.sample_triangle(instr.points, instr.sampler, instr.args)
            elif hasattr(instr, 'obj_name') and hasattr(instr, 'parameterization'):
                self.parameterize(instr)
            elif hasattr(instr, 'obj_name') and hasattr(instr, 'computation'):
                self.compute(instr)

        # Second pass: collect constraints
        for instr in self.reader.instructions:
            if hasattr(instr, 'constraint'):
                self.constraints.append(instr.constraint)

        # Optimize if enabled
        if self.optimize and self.constraints:
            self.optimize_points()

        # Convert segments
        segments = []
        for p1, p2 in self.reader.segments:
            sp1 = self.lookup_point(p1)
            sp2 = self.lookup_point(p2)
            segments.append((sp1, sp2))

        return Diagram(
            named_points=self.named_points,
            named_lines=self.named_lines,
            segments=segments,
            seg_colors=self.reader.seg_colors,
            unnamed_points=self.unnamed_points,
            unnamed_lines=self.unnamed_lines,
            ndgs={},
            goals={}
        )

    def optimize_points(self):
        """Optimize point positions to satisfy constraints"""
        # Build variable vector from all named points
        point_list = list(self.named_points.keys())
        x0 = []
        for p in point_list:
            self.point_vars[p] = len(x0)
            sp = self.named_points[p]
            x0.extend([sp.x, sp.y])

        if len(x0) == 0:
            return

        x0 = np.array(x0)

        # Define loss function
        def loss_fn(x):
            # Update points from variables
            temp_points = {}
            for p in point_list:
                idx = self.point_vars[p]
                temp_points[p] = SimplePoint(x[idx], x[idx+1])

            total_loss = 0.0

            # Evaluate each constraint
            for cons in self.constraints:
                pred = cons.pred
                args = cons.args

                # Simple constraint evaluation
                if pred == "cong":  # Equal distances
                    p1, p2, p3, p4 = [temp_points.get(a, self.named_points.get(a)) for a in args]
                    d1 = self.dist(p1, p2)
                    d2 = self.dist(p3, p4)
                    total_loss += (d1 - d2) ** 2

                elif pred == "midp":  # M is midpoint of AB
                    m, a, b = [temp_points.get(arg, self.named_points.get(arg)) for arg in args]
                    expected_m = SimplePoint((a.x + b.x)/2, (a.y + b.y)/2)
                    total_loss += (m.x - expected_m.x)**2 + (m.y - expected_m.y)**2

                elif pred == "coll":  # Three points collinear
                    p1, p2, p3 = [temp_points.get(a, self.named_points.get(a)) for a in args]
                    cross = (p2.x - p1.x)*(p3.y - p1.y) - (p2.y - p1.y)*(p3.x - p1.x)
                    total_loss += cross ** 2

                elif pred == "perp":  # Perpendicular lines
                    if len(args) == 2 and all(isinstance(a, Line) for a in args):
                        l1 = self.named_lines.get(args[0])
                        l2 = self.named_lines.get(args[1])
                        if l1 and l2:
                            (n1x, n1y), r1 = l1
                            (n2x, n2y), r2 = l2
                            dot = n1x * n2x + n1y * n2y
                            total_loss += dot ** 2

            # Regularization to keep points spread out
            for i, p1 in enumerate(point_list):
                for p2 in point_list[i+1:]:
                    idx1 = self.point_vars[p1]
                    idx2 = self.point_vars[p2]
                    d = (x[idx1] - x[idx2])**2 + (x[idx1+1] - x[idx2+1])**2
                    if d < 0.1:
                        total_loss += 10.0 * (0.1 - d)**2

            return total_loss

        # Run optimization
        result = minimize(loss_fn, x0, method='BFGS', options={'maxiter': self.n_iterations})

        # Update points with optimized values
        for p in point_list:
            idx = self.point_vars[p]
            self.named_points[p] = SimplePoint(result.x[idx], result.x[idx+1])

    def sample_triangle(self, points, sampler, args):
        """Sample a triangle"""
        if sampler == "triangle":
            # Regular triangle
            A = SimplePoint(-2, 0)
            B = SimplePoint(2, 0)
            C = SimplePoint(0, 2*math.sqrt(3))
            self.named_points[points[0]] = A
            self.named_points[points[1]] = B
            self.named_points[points[2]] = C
        elif sampler == "right-tri":
            # Right triangle
            A = SimplePoint(-2, 0)
            B = SimplePoint(2, 0)
            C = SimplePoint(2, 3)
            self.named_points[points[0]] = A
            self.named_points[points[1]] = B
            self.named_points[points[2]] = C
        elif sampler == "equi-tri":
            # Equilateral triangle
            A = SimplePoint(-2, 0)
            B = SimplePoint(2, 0)
            C = SimplePoint(0, 2*math.sqrt(3))
            self.named_points[points[0]] = A
            self.named_points[points[1]] = B
            self.named_points[points[2]] = C
        elif sampler == "iso-tri":
            # Isosceles triangle
            A = SimplePoint(0, 3)
            B = SimplePoint(-2, 0)
            C = SimplePoint(2, 0)
            self.named_points[points[0]] = A
            self.named_points[points[1]] = B
            self.named_points[points[2]] = C
        else:
            # Default triangle
            A = SimplePoint(-1.5, 0)
            B = SimplePoint(1.5, 0)
            C = SimplePoint(0, 2.5)
            self.named_points[points[0]] = A
            self.named_points[points[1]] = B
            self.named_points[points[2]] = C

    def parameterize(self, instr):
        """Parameterize a point or line"""
        obj_name = instr.obj_name
        param = instr.parameterization

        if isinstance(obj_name, Point):
            if param[0] == "coords":
                # Sample random coordinates
                x = np.random.uniform(-3, 3)
                y = np.random.uniform(-3, 3)
                self.named_points[obj_name] = SimplePoint(x, y)
            elif param[0] == "on-line":
                # Point on a line
                line = param[1][0]
                lnf = self.lookup_line(line)
                p1, p2 = self.lnf_to_points(lnf)
                t = np.random.uniform(0, 1)
                pt = SimplePoint(
                    p1.x + t * (p2.x - p1.x),
                    p1.y + t * (p2.y - p1.y)
                )
                self.named_points[obj_name] = pt
            elif param[0] == "on-seg":
                # Point on a segment
                p1 = self.lookup_point(param[1][0])
                p2 = self.lookup_point(param[1][1])
                t = np.random.uniform(0.2, 0.8)
                pt = SimplePoint(
                    p1.x + t * (p2.x - p1.x),
                    p1.y + t * (p2.y - p1.y)
                )
                self.named_points[obj_name] = pt
        elif isinstance(obj_name, Line):
            if param[0] == "line":
                # Random line
                p1 = SimplePoint(np.random.uniform(-3, 3), np.random.uniform(-3, 3))
                p2 = SimplePoint(np.random.uniform(-3, 3), np.random.uniform(-3, 3))
                lnf = self.points_to_lnf(p1, p2)
                self.named_lines[obj_name] = lnf
            elif param[0] == "through-l":
                # Line through a point
                pt = self.lookup_point(param[1][0])
                angle = np.random.uniform(0, math.pi)
                p2 = SimplePoint(pt.x + math.cos(angle), pt.y + math.sin(angle))
                lnf = self.points_to_lnf(pt, p2)
                self.named_lines[obj_name] = lnf

    def compute(self, instr):
        """Compute derived geometric objects"""
        obj_name = instr.obj_name
        computation = instr.computation

        if isinstance(obj_name, Point):
            pt = self.compute_point(computation)
            self.named_points[obj_name] = pt
        elif isinstance(obj_name, Line):
            ln = self.compute_line(computation)
            self.named_lines[obj_name] = ln

    def compute_point(self, p_comp):
        """Compute a point"""
        if isinstance(p_comp.val, FuncInfo):
            pred, args = p_comp.val
            if pred == "midp":
                p1 = self.lookup_point(args[0])
                p2 = self.lookup_point(args[1])
                return SimplePoint((p1.x + p2.x)/2, (p1.y + p2.y)/2)
            elif pred == "inter-ll":
                l1 = self.lookup_line(args[0])
                l2 = self.lookup_line(args[1])
                return self.line_intersection(l1, l2)
            elif pred == "incenter":
                A = self.lookup_point(args[0])
                B = self.lookup_point(args[1])
                C = self.lookup_point(args[2])
                return self.incenter(A, B, C)
            elif pred == "centroid":
                A = self.lookup_point(args[0])
                B = self.lookup_point(args[1])
                C = self.lookup_point(args[2])
                return SimplePoint((A.x+B.x+C.x)/3, (A.y+B.y+C.y)/3)
            elif pred == "circumcenter":
                A = self.lookup_point(args[0])
                B = self.lookup_point(args[1])
                C = self.lookup_point(args[2])
                return self.circumcenter(A, B, C)
        return SimplePoint(0, 0)

    def compute_line(self, l_comp):
        """Compute a line"""
        if isinstance(l_comp.val, FuncInfo):
            pred, args = l_comp.val
            if pred == "connecting":
                p1 = self.lookup_point(args[0])
                p2 = self.lookup_point(args[1])
                return self.points_to_lnf(p1, p2)
            elif pred == "perp-at":
                pt = self.lookup_point(args[0])
                ln = self.lookup_line(args[1])
                return self.perpendicular_at(pt, ln)
            elif pred == "para-at":
                pt = self.lookup_point(args[0])
                ln = self.lookup_line(args[1])
                return self.parallel_at(pt, ln)
            elif pred == "mediator":
                p1 = self.lookup_point(args[0])
                p2 = self.lookup_point(args[1])
                mid = SimplePoint((p1.x+p2.x)/2, (p1.y+p2.y)/2)
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                p3 = SimplePoint(mid.x - dy, mid.y + dx)
                return self.points_to_lnf(mid, p3)
        return ((1, 0), 0)

    def lookup_point(self, p):
        """Look up a point"""
        if isinstance(p.val, str):
            return self.named_points.get(p, SimplePoint(0, 0))
        elif isinstance(p.val, FuncInfo):
            return self.compute_point(p)
        return SimplePoint(0, 0)

    def lookup_line(self, l):
        """Look up a line"""
        if isinstance(l.val, str):
            return self.named_lines.get(l, ((1, 0), 0))
        elif isinstance(l.val, FuncInfo):
            return self.compute_line(l)
        return ((1, 0), 0)

    def dist(self, p1, p2):
        """Distance between two points"""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        return math.sqrt(dx*dx + dy*dy)

    def incenter(self, A, B, C):
        """Calculate incenter of triangle"""
        a = self.dist(B, C)
        b = self.dist(C, A)
        c = self.dist(A, B)
        x = (a*A.x + b*B.x + c*C.x) / (a + b + c)
        y = (a*A.y + b*B.y + c*C.y) / (a + b + c)
        return SimplePoint(x, y)

    def circumcenter(self, A, B, C):
        """Calculate circumcenter of triangle"""
        D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))
        if abs(D) < 1e-10:
            return SimplePoint((A.x+B.x+C.x)/3, (A.y+B.y+C.y)/3)

        ux = ((A.x*A.x + A.y*A.y) * (B.y - C.y) +
              (B.x*B.x + B.y*B.y) * (C.y - A.y) +
              (C.x*C.x + C.y*C.y) * (A.y - B.y)) / D
        uy = ((A.x*A.x + A.y*A.y) * (C.x - B.x) +
              (B.x*B.x + B.y*B.y) * (A.x - C.x) +
              (C.x*C.x + C.y*C.y) * (B.x - A.x)) / D
        return SimplePoint(ux, uy)

    def points_to_lnf(self, p1, p2):
        """Convert two points to line normal form (n, r)"""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = math.sqrt(dx*dx + dy*dy)
        if length < 1e-10:
            return ((1, 0), p1.x)
        nx = -dy / length
        ny = dx / length
        r = nx * p1.x + ny * p1.y
        if r < 0:
            nx, ny, r = -nx, -ny, -r
        return ((nx, ny), r)

    def lnf_to_points(self, lnf):
        """Convert line normal form to two points"""
        (nx, ny), r = lnf
        if abs(nx) < 1e-10:
            p1 = SimplePoint(0, r/ny)
            p2 = SimplePoint(1, r/ny)
        else:
            p1 = SimplePoint(r/nx, 0)
            p2 = SimplePoint(r/nx - ny, nx)
        return p1, p2

    def line_intersection(self, l1, l2):
        """Find intersection of two lines"""
        (n1x, n1y), r1 = l1
        (n2x, n2y), r2 = l2
        det = n1x * n2y - n1y * n2x
        if abs(det) < 1e-10:
            return SimplePoint(0, 0)
        x = (r1 * n2y - r2 * n1y) / det
        y = (n1x * r2 - n2x * r1) / det
        return SimplePoint(x, y)

    def perpendicular_at(self, pt, ln):
        """Line perpendicular to ln at point pt"""
        (nx, ny), r = ln
        # Perpendicular has normal (-ny, nx)
        r_new = -ny * pt.x + nx * pt.y
        if r_new < 0:
            return ((ny, -nx), -r_new)
        return ((-ny, nx), r_new)

    def parallel_at(self, pt, ln):
        """Line parallel to ln through point pt"""
        (nx, ny), r = ln
        r_new = nx * pt.x + ny * pt.y
        return ((nx, ny), r_new)


def build(lines, show_plot=True, save_plot=False, outf_prefix=None, optimize=True):
    """Build geometric diagram from GMBL instructions with optimization

    Args:
        lines: GMBL code lines
        show_plot: Show matplotlib plot
        save_plot: Save plot to file
        outf_prefix: Output file prefix
        optimize: Use scipy optimization to satisfy constraints (True) or just use fixed coordinates (False)
    """
    builder = SimpleBuilder(lines, optimize=optimize)
    diagram = builder.build()

    if show_plot or save_plot:
        fname = f"{outf_prefix}.png" if outf_prefix and save_plot else None
        diagram.plot(show=show_plot, save=save_plot, fname=fname)

    return diagram


def build_from_file(filename, show_plot=True, optimize=True):
    """Build from file with optimization"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    return build(lines, show_plot=show_plot, optimize=optimize)
