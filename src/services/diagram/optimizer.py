
from networkx import center
import torch
import torch.nn as nn
import torch.optim as optim
import random
from loguru import logger
from collections import namedtuple
from functools import partial
import math
import traceback

from .model.instructions import Parameter, Assertion
from .model.value_objects import Point, Line
from .model.entities import GeometricPoint, Diagram
from .model.types import QuadrilateralType, TriangleType, DiagramType
from src.services.diagram.initializer import Initializer

TorchPoint = namedtuple("TorchPoint", ["x", "y"])
LineSF = namedtuple("LineSF", ["a", "b", "c", "p1", "p2"])
LineNF = namedtuple("LineNF", ["n", "f"])

class Optimizer:
    def __init__(self, instructions, opts, verbosity=False):
        self.instructions = instructions
        self.opts = opts
        self.verbosity = verbosity
        self._init_state()  # Initialize all state variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _init_state(self):
        """Reset state for new optimization attempt"""
        self.name2pt = {}
        self.name2line = {}
        self.all_points = []
        self.losses = {}
        self.loss_fns = {}
        self.ndgs = {}
        self.triangles_metadata = {}
        self.circles = []
        self.quadrilaterals = []
        self.quadrilaterals_metadata = {}
        self.segments = []
        self.lines = []
        self.line_objects = {}
        self.angle_equal_assertions = []
        self.angle_measures = []  # Store angles with measure values: [(vertex, p1, p2, degrees)]
        self.perpendiculars = []  # Store perpendicular segments for rendering
        self.defined_incenters = {}  # center_name -> [triangle point names]
        self.defined_circumcenters = {}  # center_name -> [triangle point names]
        self.point_on_segment_defs = {}  # point name -> (seg_p1_name, seg_p2_name)
        self.on_circle_pairs = set()  # (point_name, center_name)
        self.tangent_specs = []  # Store tangent declarations for post-solve geometric correction
        self._tangent_spec_keys = set()
        self.unnamed_point_counter = 0
        self.has_loss = False
        self.trainable_vars = []

    def get_point(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)
        return TorchPoint(x, y)

    def mkvar(self, name, lo=-1.0, hi=1.0, init_value=None):
        if init_value is not None:
            val = torch.tensor([init_value], dtype=torch.float64, device=self.device)
        else:
            val = torch.empty(1, dtype=torch.float64, device=self.device).uniform_(lo, hi)
        param = nn.Parameter(val)
        self.trainable_vars.append(param)
        return param.squeeze()

    def generate_unnamed_point_name(self):
        """Generate sequential unnamed point names: P, P1, P2, P3..."""
        if self.unnamed_point_counter == 0:
            name = "P"
        else:
            name = f"P{self.unnamed_point_counter}"
        self.unnamed_point_counter += 1
        return name

    def const(self, x):
        return torch.tensor(x, dtype=torch.float64, device=self.device)

    def dist(self, p1: TorchPoint, p2: TorchPoint):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        return torch.sqrt(dx**2 + dy**2 + 1e-8)

    def norm(self, p: TorchPoint):
        return torch.sqrt(p.x**2 + p.y**2 + 1e-8)

    def _vector_from_points(self, pa: TorchPoint, pb: TorchPoint):
        """Calculate vector from pb to pa: (pa - pb)"""
        return (pa.x - pb.x, pa.y - pb.y)

    def _vector_length(self, vx, vy):
        """Calculate length of vector (vx, vy)"""
        return torch.sqrt(vx**2 + vy**2 + 1e-8)

    def pp2lnf(self, p1: TorchPoint, p2: TorchPoint):
        # Direction vector
        dx = p2.x - p1.x
        dy = p2.y - p1.y

        # Normal vector (perpendicular)
        n_x = -dy
        n_y = dx

        # Normalize
        n_norm = torch.sqrt(n_x**2 + n_y**2 + 1e-8)
        n_x = n_x / n_norm
        n_y = n_y / n_norm

        # Make sure normal points to upper half-plane
        if n_y < 0:
            n_x = -n_x
            n_y = -n_y

        # Distance from origin
        r = n_x * p1.x + n_y * p1.y

        n = self.get_point(n_x, n_y)
        return LineNF(n, r)

    def on_line(self, p: TorchPoint, line: LineNF):
        # line in normal form: n · p - f = 0
        return line.n.x * p.x + line.n.y * p.y - line.f

    def _point_on_segment_loss(self, point: TorchPoint, p1: TorchPoint, p2: TorchPoint):
        """Loss for point lying on segment p1-p2 (between p1 and p2)"""
        line = self.pp2lnf(p1, p2)
        on_line_loss = self.on_line(point, line)**2

        # Point between p1 and p2: point = p1 + t*(p2-p1) with 0 <= t <= 1
        vec_x = p2.x - p1.x
        vec_y = p2.y - p1.y
        point_x = point.x - p1.x
        point_y = point.y - p1.y

        vec_len_sq = vec_x**2 + vec_y**2 + 1e-8
        t = (point_x * vec_x + point_y * vec_y) / vec_len_sq

        # Strong penalty if t < 0 or t > 1 to keep point strictly between endpoints
        between_penalty = torch.relu(-t) + torch.relu(t - 1)

        return 100.0 * on_line_loss + 50.0 * between_penalty

    def _angle_bisector_equal_loss(self, p_vertex: TorchPoint, p1: TorchPoint, p2: TorchPoint, p_bisector: TorchPoint):
        """Loss for angle bisector: angle(p1, vertex, bisector) = angle(p2, vertex, bisector)"""
        cos1 = self.angle_cosine(p1, p_vertex, p_bisector)
        cos2 = self.angle_cosine(p2, p_vertex, p_bisector)
        return (cos1 - cos2)**2

    def _angle_bisector_ratio_loss(self, p_vertex: TorchPoint, p1: TorchPoint, p2: TorchPoint, p_bisector: TorchPoint):
        """Loss for angle bisector theorem"""
        # BD/DA = BC/AC (angle bisector theorem)
        ratio_bd_da = self.segment_ratio(p1, p_bisector, p_bisector, p2)
        ratio_bc_ac = self.segment_ratio(p_vertex, p1, p_vertex, p2)
        return (ratio_bd_da - ratio_bc_ac)**2

    def _angle_diff_loss(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint,
                                          p4: TorchPoint, p5: TorchPoint, p6: TorchPoint):
        """Calculate squared difference between two angles: (cos(p1-p2-p3) - cos(p4-p5-p6))^2."""
        cos1 = self.angle_cosine(p1, p2, p3)
        cos2 = self.angle_cosine(p4, p5, p6)
        return (cos1 - cos2)**2

    def _triangle_angle_difference_loss(self, pts: list, idx1: int, idx2: int):
        """Calculate difference between two angles in a triangle at given indices."""
        v1_prev = pts[(idx1-1)%3]
        v1_curr = pts[idx1]
        v1_next = pts[(idx1+1)%3]

        v2_prev = pts[(idx2-1)%3]
        v2_curr = pts[idx2]
        v2_next = pts[(idx2+1)%3]

        cos1 = self.angle_cosine(v1_prev, v1_curr, v1_next)
        cos2 = self.angle_cosine(v2_prev, v2_curr, v2_next)
        return cos1 - cos2

    def dist_to_line(self, point: TorchPoint, p1: TorchPoint, p2: TorchPoint):
        """Distance from point to line defined by p1, p2"""
        line = self.pp2lnf(p1, p2)
        return torch.abs(self.on_line(point, line))

    def centroid_loss(self, centroid: TorchPoint, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint):
        expected_x = (p1.x + p2.x + p3.x) / 3
        expected_y = (p1.y + p2.y + p3.y) / 3
        return (centroid.x - expected_x)**2 + (centroid.y - expected_y)**2

    def register_pt(self, p: TorchPoint, P, save_name=True):
        if save_name:
            if p.val in self.name2pt:
                # Duplicate point definitions appear in noisy DSL; keep first definition.
                return self.name2pt[p.val]
            self.name2pt[p.val] = P
        self.all_points.append(P)
        return P

    def register_loss(self, key, val_fn, weight: float = 1.0):
        assert key not in self.loss_fns
        self.loss_fns[key] = lambda w=weight, fn=val_fn: w * (fn() ** 2).mean()
        self.has_loss = True

    def register_ndg(self, key, val_fn, weight=1.0):
        assert key not in self.ndgs
        loss_fn = lambda w=weight, fn=val_fn: w * torch.exp(-(fn() ** 2) * 20).mean()
        self.ndgs[key] = loss_fn
        self.loss_fns[key] = loss_fn
        self.has_loss = True

    def lookup_pt(self, p):
        if isinstance(p, Point):
            if p.val in self.name2pt:
                return self.name2pt[p.val]
            else:
                raise RuntimeError(f"Point {p.val} not found")
        else:
            raise RuntimeError(f"Invalid point type: {type(p)}")

    def lookup_pt_by_name(self, name: str):
        """Lookup a point by raw point name; returns None if not found."""
        return self.name2pt.get(name)

    def _ensure_named_point(self, point_name: str):
        """Create a free point if it was referenced before being explicitly defined."""
        if point_name not in self.name2pt:
            self.sample_uniform(Point(point_name))
        return self.name2pt[point_name]


    def sample_uniform(self, p, lo=-1.0, hi=1.0, save_name=True, init_coords=None):
        if init_coords is not None:
            x = self.mkvar(f"{p.val}_x", lo, hi, init_value=init_coords[0])
            y = self.mkvar(f"{p.val}_y", lo, hi, init_value=init_coords[1])
        else:
            x = self.mkvar(f"{p.val}_x", lo, hi)
            y = self.mkvar(f"{p.val}_y", lo, hi)
        P = self.get_point(x, y)
        return self.register_pt(p, P, save_name)

    def _calculate_distance_to_line(self, point, line_p1, line_p2):
        """Calculate distance from point to line (for incircle radius)"""
        line = self.pp2lnf(line_p1, line_p2)
        return torch.abs(self.on_line(point, line))

    def _dot_product(self, pa: TorchPoint, pb: TorchPoint, pc: TorchPoint):
        """Calculate dot product of vectors (pa-pb) and (pc-pb)"""
        v1x, v1y = self._vector_from_points(pa, pb)
        v2x, v2y = self._vector_from_points(pc, pb)
        return v1x * v2x + v1y * v2y

    def _cross_product_area(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint):
        """Calculate cross product for area check (collinearity test)"""
        v1x, v1y = self._vector_from_points(p2, p1)
        v2x, v2y = self._vector_from_points(p3, p1)
        return v1x * v2y - v1y * v2x

    def perpendicular(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint, p4: TorchPoint):
        """Perpendicular constraint: (p1-p2) ⊥ (p3-p4). Returns 0 when perpendicular."""
        return (p1.x - p2.x) * (p3.x - p4.x) + (p1.y - p2.y) * (p3.y - p4.y)

    def parallel(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint, p4: TorchPoint):
        """Parallel constraint: (p1-p2) ∥ (p3-p4). Returns 0 when parallel."""
        return (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)

    def collinear(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint):
        """Collinearity constraint. Returns 0 when p1, p2, p3 are collinear."""
        return p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)

    def tangent_line_circle(self, p1: TorchPoint, p2: TorchPoint, center: TorchPoint, radius: float):
        """
        Tangent constraint: line (p1-p2) is tangent to circle (center, radius).
        Returns 0 when the distance from center to line equals radius.
        """
        # Distance from center to line p1-p2 should equal radius
        dist_to_line = self.dist_to_line(center, p1, p2)
        return dist_to_line - radius

    def vector_length(self, p1: TorchPoint, p2: TorchPoint):
        """Calculate length of vector from p1 to p2."""
        vx, vy = self._vector_from_points(p2, p1)
        return torch.sqrt(vx**2 + vy**2 + 1e-8)

    def angle_cosine(self, p1: TorchPoint, vertex: TorchPoint, p2: TorchPoint):
        """Calculate cosine of angle p1-vertex-p2."""
        dot = self._dot_product(p1, vertex, p2)
        v1_x, v1_y = self._vector_from_points(p1, vertex)
        v2_x, v2_y = self._vector_from_points(p2, vertex)

        len1 = self._vector_length(v1_x, v1_y)
        len2 = self._vector_length(v2_x, v2_y)
        return dot / (len1 * len2 + 1e-8)

    def segment_ratio(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint, p4: TorchPoint):
        """Calculate ratio of segment lengths: |p1-p2| / |p3-p4|."""
        len1 = self.vector_length(p1, p2)
        len2 = self.vector_length(p3, p4)
        return len1 / (len2 + 1e-8)

    def _sample_quadrilateral_with_init(
        self,
        points: list,
        init_method,
        *args,
        noise: float = 0.05,
        quad_type: QuadrilateralType = QuadrilateralType.GENERAL,
        metadata: dict | None = None,
    ):
        """Generic quadrilateral sampler using Initializer methods"""
        assert len(points) == 4
        init_coords = init_method(*args)
        init_coords = Initializer.add_noise(init_coords, noise)

        pt_objs = [self.sample_uniform(p, init_coords=init_coords[i]) for i, p in enumerate(points)]
        names = [p.val for p in points]
        self.quadrilaterals.append(tuple(names))
        key = tuple(names)
        # Mild non-degeneracy: keep adjacent vertices distinct to avoid overlaps
        p1, p2, p3, p4 = pt_objs
        self.register_ndg(f"quad_edge_{names[0]}_{names[1]}", lambda: self.dist(p1, p2), weight=2.0)
        self.register_ndg(f"quad_edge_{names[1]}_{names[2]}", lambda: self.dist(p2, p3), weight=2.0)
        self.register_ndg(f"quad_edge_{names[2]}_{names[3]}", lambda: self.dist(p3, p4), weight=2.0)
        self.register_ndg(f"quad_edge_{names[3]}_{names[0]}", lambda: self.dist(p4, p1), weight=2.0)
        # NOTE: Rendering uses Diagram.quadrilaterals which is populated from
        if key not in self.quadrilaterals_metadata:
            self.quadrilaterals_metadata[key] = metadata if metadata is not None else {'type': quad_type}
        return pt_objs, names

    def sample_triangle(self, points: list, constraints: dict = None):
        assert len(points) == 3

        constraints = constraints or {}
        tri_type = constraints.get('type', 'scalene')
        apex_idx = constraints.get('apex_idx', 0)
        right_idx = constraints.get('right_idx', 0)
        equal_angles = constraints.get('equal_angles')

        # Smart initialization based on type
        if tri_type == 'isosceles':
            init_coords = Initializer.init_isoceles_triangle(apex_idx)
        elif tri_type == 'right':
            init_coords = Initializer.init_right_triangle(right_idx)
        elif tri_type == 'equilateral':
            init_coords = Initializer.init_equilateral_triangle()
        elif tri_type == 'right_isosceles':
            init_coords = Initializer.init_right_isoceles_triangle(right_idx)
        elif tri_type == 'obtuse':
            init_coords = Initializer.init_obtuse_triangle(apex_idx)
        else:
            # Scalene triangle init
            init_coords = Initializer.init_scalene_triangle()

        tri_noise = 0.0 if tri_type in ('right', 'right_isosceles', 'scalene') else 0.05
        init_coords = Initializer.add_noise(init_coords, noise_scale=tri_noise)

        # Create points
        p1 = self.sample_uniform(points[0], init_coords=init_coords[0])
        p2 = self.sample_uniform(points[1], init_coords=init_coords[1])
        p3 = self.sample_uniform(points[2], init_coords=init_coords[2])
        pts = [p1, p2, p3]

        # Add geometric constraints based on type
        metadata = {'type': tri_type}

        if tri_type == 'isosceles' or tri_type == 'right_isosceles':
            apex_pt = pts[apex_idx]
            other_pts = [pts[i] for i in range(3) if i != apex_idx]
            self.register_loss(f"iso_{points[0].val}_{points[1].val}_{points[2].val}",
                              lambda ap=apex_pt, o0=other_pts[0], o1=other_pts[1]: self.dist(ap, o0) - self.dist(ap, o1),
                              weight=5000.0)

            # Keep the apex reasonably far from the base to avoid visually flat isosceles triangles.
            self.register_loss(
                f"iso_min_height_{points[0].val}_{points[1].val}_{points[2].val}",
                lambda ap=apex_pt, o0=other_pts[0], o1=other_pts[1]: torch.relu(
                    0.45 * self.dist(o0, o1) - self.dist_to_line(ap, o0, o1)
                ),
                weight=1200.0,
            )

            if tri_type == 'isosceles':
                # Keep the standard isosceles layout stable (base horizontal, apex above base).
                self.register_loss(
                    f"iso_base_horizontal_{points[0].val}_{points[1].val}_{points[2].val}",
                    lambda o0=other_pts[0], o1=other_pts[1]: o0.y - o1.y,
                    weight=2500.0,
                )
                self.register_loss(
                    f"iso_apex_above_{points[0].val}_{points[1].val}_{points[2].val}",
                    lambda ap=apex_pt, o0=other_pts[0], o1=other_pts[1]: torch.relu(
                        ((o0.y + o1.y) / 2 + 0.2 * self.dist(o0, o1)) - ap.y
                    ),
                    weight=3500.0,
                )

            others = [i for i in range(3) if i != apex_idx]
            metadata['equal_sides'] = [(apex_idx, others[0]), (apex_idx, others[1])]

        if tri_type == 'right' or tri_type == 'right_isosceles':
            right_pt = pts[right_idx]
            other_pts = [pts[i] for i in range(3) if i != right_idx]
            self.register_loss(f"right_{points[0].val}_{points[1].val}_{points[2].val}",
                              lambda: self._dot_product(other_pts[0], right_pt, other_pts[1]), weight=10.0)
            # Keep right triangles axis-aligned for stable, straight rendering.
            self.register_loss(
                f"right_axis_horizontal_{points[0].val}_{points[1].val}_{points[2].val}",
                lambda: other_pts[0].y - right_pt.y,
                weight=2500.0,
            )
            self.register_loss(
                f"right_axis_vertical_{points[0].val}_{points[1].val}_{points[2].val}",
                lambda: other_pts[1].x - right_pt.x,
                weight=2500.0,
            )
            metadata['right_angle_at'] = right_idx

        if tri_type == 'equilateral':
            self.register_loss(f"equi_12_23_{points[0].val}",
                              lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=10.0)
            self.register_loss(f"equi_23_31_{points[0].val}",
                              lambda: self.dist(p2, p3) - self.dist(p3, p1), weight=10.0)
            metadata['equal_sides'] = [(0, 1), (1, 2), (2, 0)]

        if tri_type == 'scalene':
            init_ax, _ = init_coords[0]
            init_bx, _ = init_coords[1]
            init_cx, _ = init_coords[2]
            init_mid_x = (init_bx + init_cx) / 2.0
            init_base_dx = init_cx - init_bx
            # Keep the same left/right apex bias as the initializer template.
            target_apex_offset_ratio = (init_ax - init_mid_x) / (init_base_dx + 1e-8)

            # Keep default scalene layout readable: A above, BC horizontal.
            self.register_loss(
                f"scalene_base_horizontal_{points[0].val}_{points[1].val}_{points[2].val}",
                lambda: p2.y - p3.y,
                weight=3000.0,
            )
            self.register_loss(
                f"scalene_apex_offset_{points[0].val}_{points[1].val}_{points[2].val}",
                lambda tor=target_apex_offset_ratio: (
                    (p1.x - (p2.x + p3.x) / 2.0) / (p3.x - p2.x + 1e-8) - tor
                ),
                weight=2500.0,
            )
            self.register_loss(
                f"scalene_apex_above_{points[0].val}_{points[1].val}_{points[2].val}",
                lambda: torch.relu(((p2.y + p3.y) / 2 + 0.2 * self.dist(p2, p3)) - p1.y),
                weight=4000.0,
            )
            # Acute triangle guard: each vertex should have positive dot product.
            self.register_loss(
                f"scalene_acute_{points[0].val}_{points[1].val}_{points[2].val}",
                lambda: (
                    torch.relu(-self._dot_product(p2, p1, p3))
                    + torch.relu(-self._dot_product(p1, p2, p3))
                    + torch.relu(-self._dot_product(p1, p3, p2))
                ),
                weight=1200.0,
            )

        # Handle equal_angles constraint
        if equal_angles:
            metadata['equal_angles'] = equal_angles
            for idx1, idx2 in equal_angles:
                self.register_loss(
                    f"equal_angle_{idx1}_{idx2}_{points[0].val}",
                    lambda i1=idx1, i2=idx2, p=pts: self._triangle_angle_difference_loss(p, i1, i2),
                    weight=10.0
                )

        self.register_ndg(f"tri_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda: self.collinear(p1, p2, p3), weight=20.0)
        key = (points[0].val, points[1].val, points[2].val)
        self.triangles_metadata[key] = metadata
        return [p1, p2, p3]

    #tu giac
    def sample_quadrilateral(self, points: list, quad_type: str = 'quadrilateral'):
        """Unified function to sample all types of quadrilaterals

        Args:
            points: List of 4 Point objects
            quad_type: Type of quadrilateral - 'square', 'rectangle', 'parallelogram',
                      'trapezoid', 'rhombus', or 'quadrilateral' (generic)

        Returns:
            List of 4 TorchPoint objects
        """
        assert len(points) == 4, "Quadrilateral must have exactly 4 points"

        if isinstance(quad_type, QuadrilateralType):
            quad_type_str = str(quad_type).split('.')[-1].lower()
        else:
            quad_type_str = str(quad_type).split('.')[-1].lower() if quad_type else 'quadrilateral'

        if quad_type_str == 'general':
            quad_type_str = 'quadrilateral'

        init_method = Initializer.init_quadrilateral
        init_args = (1.0,)
        init_noise = 0.05
        metadata_type = QuadrilateralType.GENERAL

        if quad_type_str == 'square':
            init_method = Initializer.init_square
            init_args = (1.0,)
            init_noise = 0.0
            metadata_type = QuadrilateralType.SQUARE
        elif quad_type_str == 'rectangle':
            init_method = Initializer.init_rectangle
            init_args = (1.6, 1.0)
            init_noise = 0.0
            metadata_type = QuadrilateralType.RECTANGLE
        elif quad_type_str == 'parallelogram':
            init_method = Initializer.init_parallelogram
            init_args = (1.3,)
            init_noise = 0.0
            metadata_type = QuadrilateralType.PARALLELOGRAM
        elif quad_type_str == 'trapezoid':
            trapezoid_style = str(self.opts.get('trapezoid_style', 'isosceles')).lower()
            init_method = Initializer.init_trapezoid_isosceles if trapezoid_style == 'isosceles' else Initializer.init_trapezoid_general
            init_args = (1.2,)
            init_noise = 0.0
            metadata_type = QuadrilateralType.TRAPEZOID
        elif quad_type_str == 'rhombus':
            init_method = Initializer.init_rhombus
            init_args = (1.0,)
            init_noise = 0.0
            metadata_type = QuadrilateralType.RHOMBUS

        pt_objs, names = self._sample_quadrilateral_with_init(
            points,
            init_method,
            *init_args,
            noise=init_noise,
            quad_type=metadata_type,
        )

        p1, p2, p3, p4 = pt_objs
        key = tuple(names)

        # Apply constraints based on quad type
        if quad_type_str == 'square':
            # Strong square constraints: equal sides + multiple right angles.
            self.register_loss(f"sq_eq_12_23_{names[0]}", lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=12.0)
            self.register_loss(f"sq_eq_23_34_{names[0]}", lambda: self.dist(p2, p3) - self.dist(p3, p4), weight=12.0)
            self.register_loss(f"sq_eq_34_41_{names[0]}", lambda: self.dist(p3, p4) - self.dist(p4, p1), weight=12.0)
            self.register_loss(f"sq_right_B_{names[0]}", lambda: self._dot_product(p1, p2, p3), weight=12.0)
            self.register_loss(f"sq_right_C_{names[0]}", lambda: self._dot_product(p2, p3, p4), weight=12.0)
            self.register_loss(f"sq_right_D_{names[0]}", lambda: self._dot_product(p3, p4, p1), weight=12.0)
            self.register_loss(f"sq_axis_horizontal_{names[0]}", lambda: p2.y - p1.y, weight=2500.0)
            self.register_loss(f"sq_axis_vertical_{names[0]}", lambda: p4.x - p1.x, weight=2500.0)
            self.register_ndg(f"sq_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)

            self.quadrilaterals_metadata[key] = {
                'type': QuadrilateralType.SQUARE,
                'equal_sides': [(0, 1), (1, 2), (2, 3), (3, 0)],
            }

        elif quad_type_str == 'rectangle':
            # Three right angles
            self.register_loss(f"rect_right_B_{names[0]}", lambda: self._dot_product(p1, p2, p3), weight=10.0)
            self.register_loss(f"rect_right_C_{names[0]}", lambda: self._dot_product(p2, p3, p4), weight=10.0)
            self.register_loss(f"rect_right_D_{names[0]}", lambda: self._dot_product(p3, p4, p1), weight=10.0)
            # Keep rectangle visually straight (axis-aligned).
            self.register_loss(f"rect_axis_horizontal_{names[0]}", lambda: p2.y - p1.y, weight=2500.0)
            self.register_loss(f"rect_axis_vertical_{names[0]}", lambda: p4.x - p1.x, weight=2500.0)
            # Encourage a non-square rectangle by keeping the aspect ratio close to the initializer.
            target_ratio = 1.6 / 1.0
            self.register_loss(
                f"rect_aspect_{names[0]}",
                lambda: self.dist(p1, p2) / (self.dist(p2, p3) + 1e-8) - target_ratio,
                weight=5.0,
            )
            self.register_ndg(f"rect_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)

            self.quadrilaterals_metadata[key] = {
                'type': QuadrilateralType.RECTANGLE,
                'equal_sides': [(0, 2), (1, 3)],
            }

        elif quad_type_str == 'parallelogram':
            # Opposite sides parallel via vector equality.
            self.register_loss(f"para_vec_x_{names[0]}", lambda: (p2.x - p1.x) - (p3.x - p4.x), weight=10.0)
            self.register_loss(f"para_vec_y_{names[0]}", lambda: (p2.y - p1.y) - (p3.y - p4.y), weight=10.0)
            # Keep adjacent sides non-perpendicular, but not overly slanted.
            target_cos = float(self.opts.get('parallelogram_target_cosine', 0.30))
            shear_weight = float(self.opts.get('parallelogram_shear_weight', 8.0))
            self.register_loss(
                f"para_shear_{names[0]}",
                lambda: self.angle_cosine(p2, p1, p4) - target_cos,
                weight=shear_weight,
            )
            # Break global rotation symmetry so the shape does not look randomly tilted.
            axis_weight = float(self.opts.get('parallelogram_axis_horizontal_weight', 900.0))
            self.register_loss(f"para_axis_horizontal_{names[0]}", lambda: p2.y - p1.y, weight=axis_weight)
            self.register_ndg(f"para_area_{names[0]}", lambda: self._cross_product_area(p2, p1, p3), weight=20.0)

            self.quadrilaterals_metadata[key] = {
                'type': QuadrilateralType.PARALLELOGRAM,
                'opposite_parallel': True,
            }

        elif quad_type_str == 'trapezoid':
            # One pair parallel
            self.register_loss(
                f"trap_para_{names[0]}",
                lambda: (p2.x - p1.x) * (p3.y - p4.y) - (p2.y - p1.y) * (p3.x - p4.x),
                weight=10.0,
            )
            self.register_ndg(f"trap_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            self.register_ndg(f"trap_ndg_top_{names[0]}", lambda: self.dist(p3, p4), weight=10.0)
            self.register_ndg(f"trap_ndg_bottom_{names[0]}", lambda: self.dist(p1, p2), weight=10.0)
            # Keep the trapezoid from collapsing into a near-collinear shape.
            self.register_loss(
                f"trap_min_height_c_{names[0]}",
                lambda: torch.relu(0.55 * self.dist(p1, p2) - self.dist_to_line(p3, p1, p2)),
                weight=4000.0,
            )
            self.register_loss(
                f"trap_min_height_d_{names[0]}",
                lambda: torch.relu(0.55 * self.dist(p1, p2) - self.dist_to_line(p4, p1, p2)),
                weight=4000.0,
            )

            self.quadrilaterals_metadata[key] = {
                'type': QuadrilateralType.TRAPEZOID,
                'parallel_sides': [(0, 1), (3, 2)],
            }

        elif quad_type_str == 'rhombus':
            # All sides equal + diagonals perpendicular
            self.register_loss(f"rhombus_eq_12_23_{names[0]}", lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=10.0)
            self.register_loss(f"rhombus_eq_23_34_{names[0]}", lambda: self.dist(p2, p3) - self.dist(p3, p4), weight=10.0)
            self.register_loss(f"rhombus_eq_34_41_{names[0]}", lambda: self.dist(p3, p4) - self.dist(p4, p1), weight=10.0)
            self.register_loss(
                f"rhombus_diag_perp_{names[0]}",
                lambda: (p3.x - p1.x) * (p4.x - p2.x) + (p3.y - p1.y) * (p4.y - p2.y),
                weight=10.0,
            )
            # Keep rhombus visually distinct from square by favoring uneven diagonal lengths.
            target_diag_ratio = 2.6
            self.register_loss(
                f"rhombus_diag_ratio_{names[0]}",
                lambda: self.dist(p1, p3) / (self.dist(p2, p4) + 1e-8) - target_diag_ratio,
                weight=8.0,
            )
            self.register_ndg(f"rhombus_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            self.register_ndg(f"rhombus_diag_ac_{names[0]}", lambda: self.dist(p1, p3), weight=10.0)

            self.quadrilaterals_metadata[key] = {
                'type': QuadrilateralType.RHOMBUS,
                'equal_sides': [(0, 1), (1, 2), (2, 3), (3, 0)],
            }

        else:
            # Only non-degeneracy constraint
            self.register_ndg(f"quad_edge_12_{names[0]}", lambda: self.dist(p1, p2), weight=8.0)
            self.register_ndg(f"quad_edge_23_{names[0]}", lambda: self.dist(p2, p3), weight=8.0)
            self.register_ndg(f"quad_edge_34_{names[0]}", lambda: self.dist(p3, p4), weight=8.0)
            self.register_ndg(f"quad_edge_41_{names[0]}", lambda: self.dist(p4, p1), weight=8.0)
            self.register_ndg(f"quad_area_123_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            self.register_ndg(f"quad_area_134_{names[0]}", lambda: self._cross_product_area(p1, p3, p4), weight=20.0)
            self.register_ndg(f"quad_area_124_{names[0]}", lambda: self._cross_product_area(p1, p2, p4), weight=14.0)
            self.register_loss(
                f"quad_convex_turn_{names[0]}",
                lambda: torch.relu(-self._cross_product_area(p1, p2, p3) * self._cross_product_area(p2, p3, p4))
                + torch.relu(-self._cross_product_area(p2, p3, p4) * self._cross_product_area(p3, p4, p1))
                + torch.relu(-self._cross_product_area(p3, p4, p1) * self._cross_product_area(p4, p1, p2)),
                weight=200.0,
            )
            self.quadrilaterals_metadata[key] = {'type': QuadrilateralType.GENERAL}

        return pt_objs


    def _define_projection(self, point_name, vertex_point, segment_points):
        assert len(segment_points) == 2

        foot = self.sample_uniform(point_name)
        vertex = self.lookup_pt(vertex_point)
        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])

        # Foot perpendicular to segment
        self.register_loss(f"perp_{point_name.val}",
                          lambda v=vertex, f=foot, a=p1, b=p2: self.perpendicular(v, f, a, b)**2,
                          weight=1000.0)  # TĂNG weight để đảm bảo vuông góc

        # Foot lies ON SEGMENT (not just collinear on the line)
        self.register_loss(f"on_seg_{point_name.val}",
                          lambda f=foot, a=p1, b=p2: self._point_on_segment_loss(f, a, b),
                          weight=5000.0)  # CỰC MẠNH - foot phải nằm trên đoạn thẳng

        # Draw helper segment from projection source to the foot (e.g., OH).
        helper_segment = (vertex_point.val, point_name.val)
        reverse_helper_segment = (point_name.val, vertex_point.val)
        if helper_segment not in self.segments and reverse_helper_segment not in self.segments:
            self.segments.append(helper_segment)
        return foot

    def _define_intersection(self, point_name, segment1_points, segment2_points):
        assert len(segment1_points) == 2 and len(segment2_points) == 2

        p1 = self.lookup_pt(segment1_points[0])
        p2 = self.lookup_pt(segment1_points[1])
        p3 = self.lookup_pt(segment2_points[0])
        p4 = self.lookup_pt(segment2_points[1])

        init_x = (p1.x.item() + p2.x.item() + p3.x.item() + p4.x.item()) / 4
        init_y = (p1.y.item() + p2.y.item() + p3.y.item() + p4.y.item()) / 4
        intersection = self.sample_uniform(point_name, init_coords=(init_x, init_y))

        # Intersection lies on both lines
        self.register_loss(f"on_line1_{point_name.val}",
                          lambda: self.collinear(intersection, p1, p2), weight=50.0)
        self.register_loss(f"on_line2_{point_name.val}",
                          lambda: self.collinear(intersection, p3, p4), weight=50.0)
        return intersection

    def _define_centroid(self, point_name, triangle_points):
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        centroid = self.sample_uniform(point_name)
        # Constraint: must be at centroid position
        def centroid_loss():
            expected_x = (p1.x + p2.x + p3.x) / 3
            expected_y = (p1.y + p2.y + p3.y) / 3
            return (centroid.x - expected_x)**2 + (centroid.y - expected_y)**2
        self.register_loss(f"centroid_{point_name.val}", centroid_loss, weight=10.0)
        return centroid

    def _define_circle_with_points(self, center_name, points_info, radius_value):
        center_coords = Initializer.init_circle_with_positioned_points(
            center=(0.0, 0.0),
            radius=radius_value,
        )
        center_coords = Initializer.add_noise(center_coords)
        center = self.sample_uniform(center_name, init_coords=center_coords[0])

        for point_name, distance, constraint_type in points_info:
            point = self.sample_uniform(point_name)

            if constraint_type == 'on_circle':
                self.register_loss(f"on_circle_{point_name.val}",
                lambda pt=point, c=center, r=radius_value:
                    (self.dist(pt, c) - r)**2, weight=200.0
            )

            elif constraint_type == 'inside':
                self.register_loss(f"inside_circle_{point_name.val}",
                lambda pt=point, c=center, d=distance:
                    (self.dist(pt, c) - d)**2, weight=150.0
            )

            elif constraint_type == 'outside':
                self.register_loss(f"outside_circle_{point_name.val}",
                lambda pt=point, c=center, d=distance:
                    (self.dist(pt, c) - d)**2, weight=150.0
            )

            else:
                self.register_loss(
                f"point_distance_{point_name.val}",
                lambda pt=point, c=center, d=distance:
                    (self.dist(pt, c) - d)**2, weight=100.0
            )
        self.circles.append((center_name.val, {
        'type': 'positioned_points',
        'radius': radius_value,
        'points': [p[0].val for p in points_info]
        }))

        return center

    def _define_incenter(self, point_name, polygon_points):
        """Define incenter for triangles and special quadrilaterals (square/rhombus)."""
        assert len(polygon_points) >= 3

        if not self._is_incircle_supported_polygon(polygon_points):
            if self.verbosity:
                point_names = [p.val for p in polygon_points]
                logger.warning(
                    f"Unsupported incenter polygon {point_names}. Only triangle, square, rhombus are supported"
                )
            return self.sample_uniform(point_name)

        self.defined_incenters[point_name.val] = [p.val for p in polygon_points]

        init_coords = Initializer.init_triangle_incircle()
        init_coords = Initializer.add_noise(init_coords)
        incenter = self.sample_uniform(point_name, init_coords=init_coords[3])

        if len(polygon_points) == 3:
            p1 = self.lookup_pt(polygon_points[0])
            p2 = self.lookup_pt(polygon_points[1])
            p3 = self.lookup_pt(polygon_points[2])
            self.register_loss(
                f"incenter_{point_name.val}",
                lambda: (self.dist_to_line(incenter, p1, p2) - self.dist_to_line(incenter, p2, p3))**2 +
                        (self.dist_to_line(incenter, p2, p3) - self.dist_to_line(incenter, p3, p1))**2,
                weight=3000.0,
            )
        else:
            point_list = [self.lookup_pt(p) for p in polygon_points]
            self._register_equal_distance_to_polygon_sides_loss(
                incenter,
                point_list,
                loss_key=f"incenter_{point_name.val}",
                weight=3000.0,
            )

        return incenter

    def _equal_radius_center_loss(self, center_point, point_list):
        """All points have equal radius from one center."""
        base_radius = self.dist(center_point, point_list[0])
        total = self.const(0.0)
        for pt in point_list[1:]:
            total = total + (base_radius - self.dist(center_point, pt))**2
        return total

    def _register_equal_radius_center_loss(self, center_point, point_list, loss_key: str, weight: float = 3000.0):
        """Register equal-radius constraint without nested function definitions."""
        if len(point_list) < 3:
            return
        if loss_key in self.loss_fns:
            return

        self.register_loss(
            loss_key,
            partial(self._equal_radius_center_loss, center_point, point_list),
            weight=weight,
        )

    def _equal_distance_to_polygon_sides_loss(self, center_point, polygon_point_list):
        """All polygon sides have equal distance from one center point."""
        if len(polygon_point_list) < 3:
            return self.const(0.0)

        base_dist = self.dist_to_line(center_point, polygon_point_list[0], polygon_point_list[1])
        total = self.const(0.0)
        side_count = len(polygon_point_list)

        for idx in range(1, side_count):
            p1 = polygon_point_list[idx]
            p2 = polygon_point_list[(idx + 1) % side_count]
            side_dist = self.dist_to_line(center_point, p1, p2)
            total = total + (base_dist - side_dist) ** 2

        return total

    def _register_equal_distance_to_polygon_sides_loss(
        self,
        center_point,
        polygon_point_list,
        loss_key: str,
        weight: float = 3000.0,
    ):
        """Register equal-distance-to-sides constraint for polygon incircle center."""
        if len(polygon_point_list) < 3:
            return
        if loss_key in self.loss_fns:
            return

        self.register_loss(
            loss_key,
            partial(self._equal_distance_to_polygon_sides_loss, center_point, polygon_point_list),
            weight=weight,
        )

    def _get_quadrilateral_metadata_by_points(self, point_names: list[str]):
        """Find quadrilateral metadata by vertex set (order-insensitive)."""
        target = set(point_names)
        for key, metadata in self.quadrilaterals_metadata.items():
            if len(key) == 4 and set(key) == target:
                return metadata
        return None

    def _is_incircle_supported_polygon(self, polygon_points):
        """Incircle is supported for triangles and special quadrilaterals (square/rhombus)."""
        if len(polygon_points) == 3:
            return True
        if len(polygon_points) != 4:
            return False

        point_names = [p.val if hasattr(p, 'val') else str(p) for p in polygon_points]
        metadata = self._get_quadrilateral_metadata_by_points(point_names)
        if not metadata:
            return False

        quad_type = metadata.get('type', '')
        quad_type_str = str(quad_type).split('.')[-1].lower()
        return quad_type_str in {'square', 'rhombus'}

    def _get_incircle_point_names(self, circle_info: dict) -> list[str]:
        """Backward-compatible accessor for incircle boundary points."""
        points = circle_info.get('points')
        if isinstance(points, list) and len(points) > 0:
            return points

        triangle = circle_info.get('triangle')
        if isinstance(triangle, list) and len(triangle) > 0:
            return triangle

        return []

    def _get_circumcircle_point_names(self, circle_info: dict) -> list[str]:
        """Backward-compatible accessor for circumcircle reference points."""
        points = circle_info.get('points')
        if isinstance(points, list) and len(points) > 0:
            return points

        triangle = circle_info.get('triangle')
        if isinstance(triangle, list) and len(triangle) > 0:
            return triangle

        return []

    def _define_circumcenter(self, point_name, polygon_points):
        """Define circumcenter for 3+ points; triangle behavior remains unchanged."""
        assert len(polygon_points) >= 3
        self.defined_circumcenters[point_name.val] = [p.val for p in polygon_points]

        p1 = self.lookup_pt(polygon_points[0])
        p2 = self.lookup_pt(polygon_points[1])
        p3 = self.lookup_pt(polygon_points[2])

        init_coords = Initializer.init_triangle_circumcircle(radius=1.0)
        init_coords = Initializer.add_noise(init_coords, noise_scale=0.02)
        circumcenter = self.sample_uniform(point_name, init_coords=init_coords[3])

        if len(polygon_points) == 3:
            self.register_loss(f"circumcenter_{point_name.val}",
                              lambda: (self.dist(circumcenter, p1) - self.dist(circumcenter, p2))**2 +
                                      (self.dist(circumcenter, p2) - self.dist(circumcenter, p3))**2,
                      weight=3000.0)
        else:
            point_list = [self.lookup_pt(p) for p in polygon_points]
            self._register_equal_radius_center_loss(
                circumcenter,
                point_list,
                loss_key=f"circumcenter_{point_name.val}",
                weight=3000.0,
            )
        return circumcenter

    def _define_orthocenter(self, point_name, triangle_points):
        """Define orthocenter - intersection of altitudes"""
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        # Create named point with smart init
        init_coords = Initializer.init_right_triangle_with_orthocenter()
        init_coords = Initializer.add_noise(init_coords)
        orthocenter = self.sample_uniform(point_name, init_coords=init_coords[3])

        # Two altitudes perpendicular to opposite sides: AH ⊥ BC, BH ⊥ AC
        self.register_loss(f"orthocenter_{point_name.val}",
                          lambda: self._dot_product(p1, orthocenter, p2, p3)**2 +
                                  self._dot_product(p2, orthocenter, p1, p3)**2,
                          weight=10.0)
        return orthocenter

    def _define_midpoint(self, point_name, segment_points):
        assert len(segment_points) == 2

        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])
        init_x = (p1.x.detach().cpu().item() + p2.x.detach().cpu().item()) / 2
        init_y = (p1.y.detach().cpu().item() + p2.y.detach().cpu().item()) / 2
        midpoint = self.sample_uniform(point_name, init_coords=(init_x, init_y))

        # Constraint 1: Midpoint position (x,y coordinates)
        self.register_loss(f"midpoint_pos_{point_name.val}",
                          lambda: (midpoint.x - (p1.x + p2.x)/2)**2 + (midpoint.y - (p1.y + p2.y)/2)**2,
                          weight=5000.0)

        # Constraint 2: Collinearity
        self.register_loss(f"midpoint_collinear_{point_name.val}",
                          lambda: self.collinear(midpoint, p1, p2),
                          weight=1000.0)

        # Constraint 3: Equal halves (|MP1| = |MP2|)
        self.register_loss(
            f"midpoint_equal_halves_{point_name.val}",
            lambda: self.dist(midpoint, p1) - self.dist(midpoint, p2),
            weight=2000.0,
        )

        return midpoint

    def _define_diameter_point(self, point_name, args):
        """Define a point as the diametrically opposite point on a circle.

        Given (diameter-point known_endpoint center):
        K = 2*Center - known_endpoint  (reflection through center)

        K is a trainable parameter initialized at the exact position.
        Constraints use .detach() on center and known_pt so gradients only
        flow to K — no interference with circumcircle/incircle constraints.
        K's target position updates each iteration as O and A converge.
        """
        assert len(args) == 2, f"diameter-point requires 2 args (known_endpoint, center), got {len(args)}"

        known_endpoint = args[0]  # e.g., A
        center_point = args[1]    # e.g., O

        known_pt = self.lookup_pt(known_endpoint)
        center = self.lookup_pt(center_point)

        known_name = known_endpoint.val
        center_name = center_point.val

        # Initialize at the exact diametrically opposite position: K = 2*O - A
        init_x = 2 * center.x.detach().cpu().item() - known_pt.x.detach().cpu().item()
        init_y = 2 * center.y.detach().cpu().item() - known_pt.y.detach().cpu().item()

        diameter_pt = self.sample_uniform(point_name, init_coords=(init_x, init_y))

        # Constraint: K = 2*O - A
        # Use .detach() on center and known_pt so gradients ONLY flow to K,
        # preventing interference with circumcircle constraints on O and A.
        # The detached values still update each iteration (current position of O and A).
        self.register_loss(f"diameter_point_reflection_{point_name.val}",
            lambda dp=diameter_pt, c=center, kp=known_pt:
                (dp.x - (2 * c.x.detach() - kp.x.detach()))**2 +
                (dp.y - (2 * c.y.detach() - kp.y.detach()))**2,
            weight=10000000.0)

        # Add segment for rendering the diameter line
        self.segments.append((known_name, point_name.val))

        logger.info(f"Defined diameter-point: {point_name.val} = 2*{center_name} - {known_name} "
                     f"at ({init_x:.4f}, {init_y:.4f})")

        return diameter_pt

    def parameter_on_seg(self, p, segment_points: list):
        assert len(segment_points) == 2
        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])
        self.point_on_segment_defs[p.val] = (segment_points[0].val, segment_points[1].val)
        # Initialize near the midpoint to help convergence
        init_x = (p1.x.detach().cpu().item() + p2.x.detach().cpu().item()) / 2
        init_y = (p1.y.detach().cpu().item() + p2.y.detach().cpu().item()) / 2
        P = self.sample_uniform(p, init_coords=(init_x, init_y))

        # IMPORTANT: Must lie on the finite segment (not just the infinite line)
        # `_point_on_segment_loss` already includes an on-line term and a strong
        # penalty for t<0 or t>1.
        self.register_loss(
            f"on_seg_{p.val}",
            lambda pt=P, a=p1, b=p2: self._point_on_segment_loss(pt, a, b),
            weight=500.0,
        )
        return P

    def parameter_on_line(self, p, line_points):
        assert len(line_points) == 2
        p1 = self.lookup_pt(line_points[0])
        p2 = self.lookup_pt(line_points[1])

        # Simple initialization without analytical solution
        P = self.sample_uniform(p, save_name=False)

        def on_line_loss():
            line = self.pp2lnf(p1, p2)
            return self.on_line(P, line)**2
        self.register_loss(f"on_line_{p.val}", on_line_loss, weight=5.0)  # Giảm xuống 5
        return self.register_pt(p, P)

    def _define_line_intersection(self, point_name, line1_points, line2_points):
        assert len(line1_points) == 2 and len(line2_points) == 2
        p1 = self.lookup_pt(line1_points[0])
        p2 = self.lookup_pt(line1_points[1])
        p3 = self.lookup_pt(line2_points[0])
        p4 = self.lookup_pt(line2_points[1])
        intersection = self.sample_uniform(point_name)

        self.register_loss(f"on_line1_{point_name.val}",
                          lambda: self.collinear(intersection, p1, p2), weight=10.0)
        self.register_loss(f"on_line2_{point_name.val}",
                          lambda: self.collinear(intersection, p3, p4), weight=10.0)
        return intersection

    def _define_angle_bisector(self, point_name, angle_points):
        assert len(angle_points) >= 3, "angle_bisector requires 3 points [B, A, C] where A is vertex"

        # DSL: (bisector B A C)
        p1 = self.lookup_pt(angle_points[0])  # B
        vertex = self.lookup_pt(angle_points[1])  # A (vertex - đỉnh góc)
        p2 = self.lookup_pt(angle_points[2])  # C

        init_x = (vertex.x.item() + p1.x.item() + p2.x.item()) / 3
        init_y = (vertex.y.item() + p1.y.item() + p2.y.item()) / 3
        bisector_point = self.sample_uniform(point_name, init_coords=(init_x, init_y))

        # Save metadata for rendering
        if not hasattr(self, 'angle_bisectors_metadata'):
            self.angle_bisectors_metadata = []

        self.angle_bisectors_metadata.append({
            'vertex': angle_points[1].val,  # A is the vertex
            'bisector_point': point_name.val,
            'angle_points': [p.val for p in angle_points]
        })
        self._process_angle_bisector(angle_points[1], point_name, angle_points)
        return bisector_point

    def _define_perpendicular_bisector_point(self, point_name, segment_points):
        """Define a point that lies on the perpendicular bisector of a segment"""
        assert len(segment_points) == 2

        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])

        # Initialize near the midpoint to improve convergence for bisector constraints.
        init_x = (p1.x.detach().cpu().item() + p2.x.detach().cpu().item()) / 2
        init_y = (p1.y.detach().cpu().item() + p2.y.detach().cpu().item()) / 2
        point = self.sample_uniform(point_name, init_coords=(init_x, init_y))

        # 1) Locus condition: points on the perpendicular bisector satisfy |PP1| = |PP2|.
        self.register_loss(
            f"perp_bisector_equal_dist_{point_name.val}",
            lambda: self.dist(point, p1) - self.dist(point, p2),
            weight=3000.0,
        )

        # 2) Direction condition: vector midpoint->P is perpendicular to segment P1P2.
        self.register_loss(
            f"perp_bisector_perp_dir_{point_name.val}",
            lambda: ((point.x - (p1.x + p2.x) / 2) * (p2.x - p1.x)
                     + (point.y - (p1.y + p2.y) / 2) * (p2.y - p1.y)),
            weight=3000.0,
        )

        # 3) Keep base segment non-degenerate for meaningful perpendicular-bisector geometry.
        ndg_key = f"perp_bisector_base_ndg_{segment_points[0].val}_{segment_points[1].val}"
        if ndg_key not in self.ndgs:
            self.register_ndg(
                ndg_key,
                lambda: self.dist(p1, p2),
                weight=10.0,
            )
        return point

    def process_instruction(self, instr):
        if isinstance(instr, Parameter):
            self.process_parameter(instr)
        elif isinstance(instr, Assertion):
            self.process_assertion(instr)

    def process_parameter(self, instr):
        diagram_type = instr.diagram_type
        param_type = instr.param_type
        objects = instr.objects
        args = instr.args

        if diagram_type == DiagramType.TRIANGLE:
            self._process_triangle_parameter(param_type, objects, args)
        elif diagram_type == DiagramType.QUADRILATERAL:
            self._process_quadrilateral_parameter(param_type, objects, args)
        elif diagram_type == DiagramType.POINT:
            self._process_point_parameter(param_type, objects, args)
        elif diagram_type == DiagramType.CIRCLE:
            self._process_circle_parameter(param_type, objects, args)
        elif diagram_type == DiagramType.SEGMENT:
            self._process_segment_parameter(objects)
        elif diagram_type == DiagramType.LINE:
            self._process_line_parameter(param_type, objects, args)
        else:
            if self.verbosity:
                logger.warning(f"Unsupported diagram type: {diagram_type}")

    def _process_angle_bisector(self, vertex, bisector_point, angle_points):
        # angle_points = [B, A, C]
        p1 = self.name2pt[angle_points[0].val]  # B
        p_vertex = self.name2pt[vertex.val]  # A (đỉnh)
        p2 = self.name2pt[angle_points[2].val]  # C
        p_bisector = self.name2pt[bisector_point.val]  # M (điểm trên phân giác)

        key = f"{vertex.val}_{bisector_point.val}"
        # Bisector constraint: Equal angles
        self.register_loss(
            f"bisector_equal_angle_{key}",
            lambda pv=p_vertex, p_1=p1, p_2=p2, pb=p_bisector: self._angle_bisector_equal_loss(pv, p_1, p_2, pb),
            weight=100.0
        )

    def _process_triangle_parameter(self, param_type, objects, args):
        if isinstance(param_type, TriangleType):
            param_type_str = str(param_type).split('.')[-1].lower()
        else:
            param_type_str = str(param_type).lower() if param_type else ""

        constraints = {}
        if param_type_str == "isosceles":
            constraints['type'] = 'isosceles'
            if args:
                # Find apex index
                for i, obj in enumerate(objects):
                    if obj.val == args[0].val:
                        constraints['apex_idx'] = i
                        break
        elif param_type_str in ["right"]:
            constraints['type'] = 'right'
            if args:
                # Find right angle vertex index
                for i, obj in enumerate(objects):
                    if obj.val == args[0].val:
                        constraints['right_idx'] = i
                        break
        elif param_type_str in ["equilateral", "equi"]:
            constraints['type'] = 'equilateral'
        elif param_type_str in ["right_isosceles", "right-isosceles"]:
            constraints['type'] = 'right_isosceles'
            if args:
                for i, obj in enumerate(objects):
                    if obj.val == args[0].val:
                        constraints['right_idx'] = i
                        constraints['apex_idx'] = i
                        break
        else:
            constraints['type'] = 'scalene'
        self.sample_triangle(objects, constraints)

    def _process_quadrilateral_parameter(self, param_type, objects, args):
        """Process quadrilateral parameters - unified handler for all quadrilateral types"""
        if param_type:
            param_type_str = str(param_type).split('.')[-1].lower()
        else:
            param_type_str = "quadrilateral"

        if param_type_str == "general":
            param_type_str = "quadrilateral"

        valid_quad_types = {
            "square",
            "rectangle",
            "parallelogram",
            "trapezoid",
            "rhombus",
            "quadrilateral",
        }
        if param_type_str not in valid_quad_types:
            if self.verbosity:
                logger.warning(f"Unknown quadrilateral type '{param_type_str}', fallback to generic quadrilateral")
            param_type_str = "quadrilateral"

        self.sample_quadrilateral(objects, quad_type=param_type_str)

        logger.info(f"Processed quadrilateral type: {param_type_str}")

    def _process_point_parameter(self, param_type, objects, args):
        param_type_str = str(param_type).lower() if param_type else ""

        # No-op duplicate free-point definitions, e.g. `(define A point)` after `(triangle (A B C))`.
        if objects and param_type_str in ["coords", ""] and objects[0].val in self.name2pt:
            if self.verbosity:
                logger.warning(f"Skipping duplicate free-point define for {objects[0].val}")
            return

        if param_type_str == "centroid":
            self._define_centroid(objects[0], args)
        elif param_type_str == "orthocenter":
            self._define_orthocenter(objects[0], args)
        elif param_type_str == "incenter":
            self._define_incenter(objects[0], args)
        elif param_type_str == "circumcenter":
            self._define_circumcenter(objects[0], args)
        elif param_type_str == "midpoint":
            self._define_midpoint(objects[0], args)
        elif param_type_str == "projection":
            self._define_projection(objects[0], args[0], args[1:])
        elif param_type_str == "segment":
            self.parameter_on_seg(objects[0], args)
        elif param_type_str == "line":
            self.parameter_on_line(objects[0], args)
        elif param_type_str in ["inter-ll", "inter_ll"]:
            # args should be 4 points: line1_p1, line1_p2, line2_p1, line2_p2
            if len(args) >= 4:
                self._define_line_intersection(objects[0], args[0:2], args[2:4])
            else:
                if self.verbosity:
                    logger.warning(f"inter-ll requires 4 points, got {len(args)}")
        elif param_type_str in ["perp-bisector", "perpendicular-bisector"]:
            self._define_perpendicular_bisector_point(objects[0], args)
        elif param_type_str == "bisector":
            self._define_angle_bisector(objects[0], args)
        elif param_type_str == "diameter-point":
            # args = (known_endpoint, center)
            # Define point as reflection of known_endpoint through center: K = 2*Center - A
            self._define_diameter_point(objects[0], args)
        elif param_type_str == "coords" or param_type_str == "":
            self.sample_uniform(objects[0])
        else:
            if self.verbosity:
                logger.warning(f"Unsupported point construction: {param_type_str}")

    def _process_circle_parameter(self, param_type, objects, args):
        """Process circle instructions and track them"""
        param_type_str = str(param_type).lower() if param_type else ""
        center_name = objects[0].val if hasattr(objects[0], 'val') else str(objects[0])

        logger.info(f"Processing circle: center={center_name}, param_type={param_type_str}, args={args}")

        if param_type_str == "incircle":
            point_names = [p.val for p in args]

            if not self._is_incircle_supported_polygon(args):
                if self.verbosity:
                    logger.warning(
                        f"Skipping incircle {center_name}: only triangle, square, rhombus are supported, got {point_names}"
                    )
                return

            if len(point_names) == 3:
                self.circles.append((center_name, {'type': 'incircle', 'triangle': point_names}))
            else:
                self.circles.append((center_name, {'type': 'incircle', 'points': point_names}))

            if len(point_names) == 4:
                center_point = self._ensure_named_point(center_name)
                point_list = [self._ensure_named_point(name) for name in point_names]
                joined_names = "_".join(point_names)
                self._register_equal_distance_to_polygon_sides_loss(
                    center_point,
                    point_list,
                    loss_key=f"incircle_{center_name}_{joined_names}",
                    weight=2500.0,
                )

            if center_name not in self.defined_incenters:
                self.defined_incenters[center_name] = point_names
        elif param_type_str == "circumcircle":
            point_names = [p.val for p in args]
            if len(point_names) == 3:
                # Keep legacy format for triangle circumcircle.
                self.circles.append((center_name, {'type': 'circumcircle', 'triangle': point_names}))
            else:
                # Use explicit multi-point format for quadrilateral and higher.
                self.circles.append((center_name, {'type': 'circumcircle', 'points': point_names}))

            if len(point_names) >= 3:
                center_point = self._ensure_named_point(center_name)
                point_list = [self._ensure_named_point(name) for name in point_names]
                joined_names = "_".join(point_names)
                self._register_equal_radius_center_loss(
                    center_point,
                    point_list,
                    loss_key=f"circumcircle_{center_name}_{joined_names}",
                    weight=2500.0,
                )

                if center_name not in self.defined_circumcenters:
                    self.defined_circumcenters[center_name] = point_names
        elif param_type_str == "auto":
            if center_name in self.defined_circumcenters:
                circum_points = self.defined_circumcenters[center_name]
                if len(circum_points) == 3:
                    self.circles.append((center_name, {
                        'type': 'circumcircle',
                        'triangle': circum_points
                    }))
                else:
                    self.circles.append((center_name, {
                        'type': 'circumcircle',
                        'points': circum_points
                    }))
            elif center_name in self.defined_incenters:
                incircle_points = self.defined_incenters[center_name]
                if len(incircle_points) == 3:
                    self.circles.append((center_name, {
                        'type': 'incircle',
                        'triangle': incircle_points
                    }))
                elif len(incircle_points) == 4:
                    self.circles.append((center_name, {
                        'type': 'incircle',
                        'points': incircle_points
                    }))
                else:
                    if self.verbosity:
                        logger.warning(
                            f"Skipping auto-incircle for {center_name}: unsupported boundary points {incircle_points}"
                        )
            else:
                # Fallback to default positioned circle when no center metadata exists.
                self.circles.append((center_name, {
                    'type': 'positioned',
                    'radius': 1.0
                }))
        elif param_type_str == "radius":
            # Circle with explicit radius: (circle O (radius 0.05))
            radius = float(args[0].val) if args and hasattr(args[0], 'val') else float(args[0]) if args else 0.5
            self.circles.append((center_name, {
                'type': 'positioned',
                'radius': radius
            }))
        elif param_type_str == "positioned" or param_type_str == "with_points":
            radius = float(args[0].val) if args else 0.5
            point_names = [p.val for p in args[1:]] if len(args) > 1 else []
            self.circles.append((center_name, {
            'type': 'positioned',
            'radius': radius,
            'points': point_names
        }))
        else:
            if self.verbosity:
                logger.warning(f"Unsupported circle type: {param_type_str}")

    def _process_segment_parameter(self, objects):
        """Track segment for visualization"""
        if len(objects) >= 2:
            p1_name = objects[0].val if hasattr(objects[0], 'val') else str(objects[0])
            p2_name = objects[1].val if hasattr(objects[1], 'val') else str(objects[1])
            # Support minimal DSL like `(segment A B)` by auto-creating endpoints.
            self._ensure_named_point(p1_name)
            self._ensure_named_point(p2_name)
            self.segments.append((p1_name, p2_name))

    def _process_line_parameter(self, param_type, objects, args):
        """Process line instructions - store for visualization"""
        # Line through 2 points: (line A B)
        if len(objects) >= 2:
            p1_name = objects[0].val if hasattr(objects[0], 'val') else str(objects[0])
            p2_name = objects[1].val if hasattr(objects[1], 'val') else str(objects[1])
            self._ensure_named_point(p1_name)
            self._ensure_named_point(p2_name)
            self.lines.append((p1_name, p2_name))

    def process_assertion(self, assertion):
        """Process assertion/constraint instructions"""
        # Assertions are handled separately - they add constraints to existing objects
        if self.verbosity:
            logger.info(f"Processing assertion: {assertion}")

        if hasattr(assertion, 'constraint_type'):
            if assertion.constraint_type == 'parallel':
                self._add_parallel_constraint(assertion.objects)
            elif assertion.constraint_type == 'perpendicular':
                self._add_perpendicular_constraint(assertion.objects)
            elif assertion.constraint_type == 'angle_equal':
                self._add_angle_equal_constraint(assertion.objects)
            elif assertion.constraint_type == 'angle_measure':
                self._add_angle_measure(assertion.objects)
            elif assertion.constraint_type == 'on_segment':
                self._add_on_segment_constraint(assertion.objects)
            elif assertion.constraint_type == 'distance':
                self._add_distance_constraint(assertion.objects)
            elif assertion.constraint_type == 'equal_distance':
                self._add_equal_distance_constraint(assertion.objects)
            elif assertion.constraint_type == 'fixed_distance':
                # fixed distance constraint (A B distance_value)
                self._add_fixed_distance_constraint(assertion.objects, assertion.distance)
            elif assertion.constraint_type == 'on_circle':
                self._add_on_circle_constraint(assertion.objects)
                # Sau khi thêm on-circle constraint, enforce góc tối thiểu
                self._enforce_minimum_angle_between_circle_points()
            elif assertion.constraint_type == 'diameter':
                # Check if center already has a circle defined
                center_obj = assertion.objects[2]
                center_name = center_obj.val
                has_existing_circle = any(cn == center_name for cn, _ in self.circles)
                if has_existing_circle:
                    self._add_diameter_on_existing_circle(assertion.objects)
                else:
                    self._add_diameter_constraint(assertion.objects)
            elif assertion.constraint_type == 'tangent':
                self._add_tangent_constraint(assertion.objects)
            elif assertion.constraint_type == 'diameter_collinear':
                self._add_diameter_collinear_constraint(assertion.objects)

    def _add_diameter_constraint(self, objects: list):
        """Standalone diameter: creates a new circle and positions center at midpoint.

        Uses MODERATE weights (5000) so constraints cooperate with (not overpower)
        parallelogram/triangle/other shape constraints.
        Full gradient flow to all points — needed for standalone free-point diameter.
        """
        assert len(objects) == 3, "Diameter constraint requires 3 points: endpoint1, endpoint2, center"

        p1_obj = objects[0]
        p2_obj = objects[1]
        center_obj = objects[2]

        try:
            p1 = self.lookup_pt(p1_obj)
            p2 = self.lookup_pt(p2_obj)
            center = self.lookup_pt(center_obj)

            p1_name = p1_obj.val
            p2_name = p2_obj.val
            center_name = center_obj.val

            # Diameter implies a circle centered at center_name even if DSL omitted `(circle center)`.
            if not any(circle_name == center_name for circle_name, _ in self.circles):
                self.circles.append((center_name, {
                    'type': 'diameter',
                    'endpoints': [p1_name, p2_name]
                }))

            # 1. Both points on circle
            self.register_loss(f"diameter_on_circle_{p1_name}_{center_name}",
                lambda pt=p1, c=center, cn=center_name:
                    (self.dist(pt, c) - self._get_circle_radius(cn))**2,
                weight=5000.0)

            self.register_loss(f"diameter_on_circle_{p2_name}_{center_name}",
                lambda pt=p2, c=center, cn=center_name:
                    (self.dist(pt, c) - self._get_circle_radius(cn))**2,
                weight=5000.0)

            # 2. Center is midpoint
            self.register_loss(f"diameter_midpoint_{center_name}_{p1_name}_{p2_name}",
                lambda c=center, pt1=p1, pt2=p2:
                    (c.x - (pt1.x + pt2.x)/2)**2 + (c.y - (pt1.y + pt2.y)/2)**2,
                weight=5000.0)

            # 3. Collinear
            self.register_loss(f"diameter_collinear_{center_name}_{p1_name}_{p2_name}",
                lambda c=center, pt1=p1, pt2=p2:
                    self.collinear(c, pt1, pt2),
                weight=5000.0)

            # Add segment for rendering the diameter line
            seg = (p1_name, p2_name)
            rev_seg = (p2_name, p1_name)
            if seg not in self.segments and rev_seg not in self.segments:
                self.segments.append(seg)

            logger.info(f"Added diameter constraint: {p1_name}{p2_name} diameter of circle {center_name}")
        except Exception as e:
            logger.warning(f"Failed to add diameter constraint: {e}")

    def _add_diameter_on_existing_circle(self, objects: list):
        """Add diameter constraint when the center already has a circle defined.
        This does NOT create a new circle entry — it reuses the existing one
        and only adds geometric constraints (on-circle, midpoint, collinear).
        Endpoints are auto-defined if they haven't been created yet,
        initialized at the diametrically opposite position for fast convergence."""
        assert len(objects) == 3, "Diameter constraint requires 3 points: endpoint1, endpoint2, center"

        p1_obj = objects[0]
        p2_obj = objects[1]
        center_obj = objects[2]

        try:
            p1_name = p1_obj.val
            p2_name = p2_obj.val
            center_name = center_obj.val

            center = self.lookup_pt(center_obj)

            # Auto-define endpoints if they haven't been defined yet
            # When one endpoint exists, place the other at the diametrically opposite position
            p1 = self.lookup_pt_by_name(p1_name)
            p2 = self.lookup_pt_by_name(p2_name)

            if p1 is None and p2 is not None:
                # p2 exists, place p1 at diametrically opposite position: p1 = 2*center - p2
                init_x = 2 * center.x.item() - p2.x.item()
                init_y = 2 * center.y.item() - p2.y.item()
                logger.info(f"Diameter (existing circle): auto-defining {p1_name} opposite to {p2_name} at ({init_x:.4f}, {init_y:.4f})")
                p1 = self.sample_uniform(Point(p1_name), init_coords=(init_x, init_y))
            elif p1 is None:
                logger.info(f"Diameter (existing circle): auto-defining endpoint {p1_name}")
                p1 = self._ensure_named_point(p1_name)

            if p2 is None and p1 is not None:
                # p1 exists, place p2 at diametrically opposite position: p2 = 2*center - p1
                init_x = 2 * center.x.item() - p1.x.item()
                init_y = 2 * center.y.item() - p1.y.item()
                logger.info(f"Diameter (existing circle): auto-defining {p2_name} opposite to {p1_name} at ({init_x:.4f}, {init_y:.4f})")
                p2 = self.sample_uniform(Point(p2_name), init_coords=(init_x, init_y))
            elif p2 is None:
                logger.info(f"Diameter (existing circle): auto-defining endpoint {p2_name}")
                p2 = self._ensure_named_point(p2_name)

            # Center is midpoint of the two endpoints.
            # .detach() on endpoints to prevent gradient interference with shape constraints.
            self.register_loss(f"diameter_midpoint_{center_name}_{p1_name}_{p2_name}",
                lambda c=center, pt1=p1, pt2=p2:
                    (c.x - (pt1.x.detach() + pt2.x.detach())/2)**2 +
                    (c.y - (pt1.y.detach() + pt2.y.detach())/2)**2,
                weight=10000000.0)

            # Three points are collinear.
            # .detach() on endpoints to prevent gradient interference.
            self.register_loss(f"diameter_collinear_{center_name}_{p1_name}_{p2_name}",
                lambda c=center, pt1=p1, pt2=p2:
                    ((pt2.y.detach() - pt1.y.detach()) * (c.x - pt1.x.detach()) -
                     (c.y - pt1.y.detach()) * (pt2.x.detach() - pt1.x.detach())),
                weight=10000000.0)

            # Add segment for rendering the diameter line
            self.segments.append((p1_name, p2_name))

            logger.info(f"Added diameter-on-existing-circle: {p1_name}{p2_name} diameter of existing circle {center_name}")
        except Exception as e:
            logger.warning(f"Failed to add diameter on existing circle: {e}")

    def _add_diameter_collinear_constraint(self, objects: list):
        """Lightweight diameter constraint for when both endpoints are already defined.

        Only adds collinearity (Center, P1, P2 on same line) with moderate weight.
        Combined with circumcircle (OA=OB=OC), collinearity alone guarantees
        O is the midpoint of P1P2, making P1P2 a diameter.

        This avoids the weight conflicts that heavy midpoint/on-circle constraints cause.
        """
        assert len(objects) == 3, "diameter_collinear requires 3 points: center, p1, p2"

        center_obj = objects[0]
        p1_obj = objects[1]
        p2_obj = objects[2]

        try:
            center = self.lookup_pt(center_obj)
            p1 = self.lookup_pt(p1_obj)
            p2 = self.lookup_pt(p2_obj)

            center_name = center_obj.val
            p1_name = p1_obj.val
            p2_name = p2_obj.val

            # Collinear constraint with weight comparable to circumcircle (2500)
            # so they cooperate rather than compete
            self.register_loss(f"diameter_collinear_{center_name}_{p1_name}_{p2_name}",
                lambda c=center, pt1=p1, pt2=p2: self.collinear(c, pt1, pt2),
                weight=2500.0)

            # Add segment for rendering the diameter line
            seg = (p1_name, p2_name)
            rev_seg = (p2_name, p1_name)
            if seg not in self.segments and rev_seg not in self.segments:
                self.segments.append(seg)

            logger.info(f"Added diameter_collinear: {center_name} collinear with {p1_name}{p2_name} (weight=2500)")
        except Exception as e:
            logger.warning(f"Failed to add diameter_collinear constraint: {e}")

    def _get_circle_radius(self, center_name):
        """Get radius expression of a circle by center name (supports positioned/incircle/circumcircle)."""
        for circle_name, circle_info in self.circles:
            if circle_name == center_name:
                circle_type = circle_info.get('type')

                if circle_type == 'incircle':
                    boundary = self._get_incircle_point_names(circle_info)
                    if len(boundary) >= 2 and all(name in self.name2pt for name in [center_name, boundary[0], boundary[1]]):
                        center = self.lookup_pt_by_name(center_name)
                        p1 = self.lookup_pt_by_name(boundary[0])
                        p2 = self.lookup_pt_by_name(boundary[1])
                        if center is not None and p1 is not None and p2 is not None:
                            return self.dist_to_line(center, p1, p2)

                if circle_type == 'circumcircle':
                    circum_points = self._get_circumcircle_point_names(circle_info)
                    if len(circum_points) >= 1 and all(name in self.name2pt for name in [center_name, circum_points[0]]):
                        center = self.lookup_pt_by_name(center_name)
                        p1 = self.lookup_pt_by_name(circum_points[0])
                        if center is not None and p1 is not None:
                            return self.dist(center, p1)

                if circle_type == 'diameter':
                    endpoints = circle_info.get('endpoints', [])
                    if len(endpoints) >= 1 and all(name in self.name2pt for name in [center_name, endpoints[0]]):
                        center = self.lookup_pt_by_name(center_name)
                        p1 = self.lookup_pt_by_name(endpoints[0])
                        if center is not None and p1 is not None:
                            return self.dist(center, p1)

                return self.const(circle_info.get('radius', 1.0))
        return self.const(1.0)


    def _add_angle_measure(self, points: list):
        """Store angle measure for display and enforce geometric angle value."""
        # DSL: (angle-measure A C B 110)
        if len(points) < 4:
            logger.warning(f"Angle-measure needs 4 values (3 points + degrees), got {len(points)}")
            return

        p1_name = points[0].val  # A
        vertex_name = points[1].val  # C (đỉnh góc)
        p2_name = points[2].val  # B
        degrees = float(points[3].val) if hasattr(points[3], 'val') else float(points[3])  # 110

        # Enforce angle value using cosine target.
        p1 = self.lookup_pt(points[0])
        vertex = self.lookup_pt(points[1])
        p2 = self.lookup_pt(points[2])
        target_cos = math.cos(math.radians(degrees))
        self.register_loss(
            f"angle_measure_{p1_name}_{vertex_name}_{p2_name}_{int(round(degrees))}",
            lambda pa=p1, pv=vertex, pb=p2, t=target_cos: self.angle_cosine(pa, pv, pb) - t,
            weight=5000.0,
        )

        # Store for later rendering
        self.angle_measures.append((vertex_name, p1_name, p2_name, degrees))
        if self.verbosity:
            logger.info(f"Added angle measure: angle {p1_name}{vertex_name}{p2_name} = {degrees}°")

    def _add_on_segment_constraint(self, points: list):
        """Add constraint: point lies on segment"""
        if len(points) != 3:
            logger.warning(f"on_segment constraint needs 3 points (point, seg_p1, seg_p2), got {len(points)}")
            return

        point = self.lookup_pt(points[0])
        seg_p1 = self.lookup_pt(points[1])
        seg_p2 = self.lookup_pt(points[2])

        point_name = points[0].val
        p1_name = points[1].val
        p2_name = points[2].val

        # Check if this is a center-on-diameter case (center on chord of its own circle)
        is_diameter = False
        for circle_name, circle_info in self.circles:
            if circle_name == point_name:
                # Check if both p1 and p2 are on this circle
                p1_on_circle = any(f"on_circle_{p1_name}_{circle_name}" in key for key in self.loss_fns.keys())
                p2_on_circle = any(f"on_circle_{p2_name}_{circle_name}" in key for key in self.loss_fns.keys())

                if p1_on_circle and p2_on_circle:
                    is_diameter = True
                    if self.verbosity:
                        logger.info(f"Detected DIAMETER: {p1_name}{p2_name} passes through center {point_name}")
                    break

        if is_diameter:
            # For diameter: center MUST be midpoint
            self.register_loss(f"diameter_midpoint_{point_name}_{p1_name}_{p2_name}",
                lambda pt=point, p1=seg_p1, p2=seg_p2:
                    (pt.x - (p1.x + p2.x)/2)**2 + (pt.y - (p1.y + p2.y)/2)**2,
                weight=1000000.0
            )
            # Ensure collinearity
            self.register_loss(f"diameter_collinear_{point_name}_{p1_name}_{p2_name}",
                lambda pt=point, p1=seg_p1, p2=seg_p2: self.collinear(pt, p1, p2)**2,
                weight=1000000.0
            )
        else:
            # Normal on-segment constraint
            key = f"{point_name}_on_{p1_name}{p2_name}"
            self.register_loss(f"on_segment_{key}",
                lambda pt=point, p1=seg_p1, p2=seg_p2: self._point_on_segment_loss(pt, p1, p2),
                weight=50000.0
            )

        if self.verbosity:
            logger.info(f"Added on-segment constraint: {point_name} on {p1_name}{p2_name} (diameter={is_diameter})")

    def _add_distance_constraint(self, points: list):
        if len(points) != 3:
            logger.warning(f"Distance constraint needs 3 points (point1, point2, distance), got {len(points)}")
            return

        p1 = self.lookup_pt(points[0])
        p2 = self.lookup_pt(points[1])
        distance_value = float(points[2].val) if hasattr(points[2], 'val') else float(points[2])

        self.register_loss(
            f"distance_{points[0].val}_{points[1].val}",
            lambda pt1=p1, pt2=p2, d=distance_value: (self.dist(pt1, pt2) - d)**2,
            weight=100.0
        )

    def _add_equal_distance_constraint(self, points: list):
        """Add constraint that distance(p1, p2) = distance(p3, p4)"""
        if len(points) != 4:
            logger.warning(f"Equal-distance constraint needs 4 points (p1, p2, p3, p4), got {len(points)}")
            return

        point_names = [pt.val for pt in points]
        p1 = self.lookup_pt(points[0])
        p2 = self.lookup_pt(points[1])
        p3 = self.lookup_pt(points[2])
        p4 = self.lookup_pt(points[3])

        eq_key = f"equal_distance_{point_names[0]}{point_names[1]}_{point_names[2]}{point_names[3]}"
        if eq_key not in self.loss_fns:
            self.register_loss(
                eq_key,
                lambda pt1=p1, pt2=p2, pt3=p3, pt4=p4: self.dist(pt1, pt2) - self.dist(pt3, pt4),
                weight=3000.0  # Tăng lên để đảm bảo AC = AD chính xác hơn
            )

        # Add non-degeneracy constraints: ensure segments are not degenerate (points not coincident)
        # This prevents solutions where p1=p2 or p3=p4
        min_segment_length = 0.15  # Minimum segment length

        ndg_key_1 = f"ndg_segment_{point_names[0]}{point_names[1]}"
        if ndg_key_1 not in self.ndgs:
            self.register_ndg(
                ndg_key_1,
                lambda pt1=p1, pt2=p2: self.dist(pt1, pt2),
                weight=10.0
            )

        ndg_key_2 = f"ndg_segment_{point_names[2]}{point_names[3]}"
        if ndg_key_2 not in self.ndgs:
            self.register_ndg(
                ndg_key_2,
                lambda pt3=p3, pt4=p4: self.dist(pt3, pt4),
                weight=10.0
            )

        # If equal-distance describes two sides that share one triangle vertex (e.g. AB = AC),
        # keep the opposite side from collapsing to a tiny segment.
        shared_vertex = None
        side_end_1 = None
        side_end_2 = None
        shared_patterns = (
            (0, 2, 1, 3),
            (0, 3, 1, 2),
            (1, 2, 0, 3),
            (1, 3, 0, 2),
        )

        for shared_idx_1, shared_idx_2, end_idx_1, end_idx_2 in shared_patterns:
            if (
                point_names[shared_idx_1] == point_names[shared_idx_2]
                and point_names[end_idx_1] != point_names[end_idx_2]
            ):
                candidate_set = {point_names[shared_idx_1], point_names[end_idx_1], point_names[end_idx_2]}
                is_triangle_side_pair = any(
                    len(tri_key) == 3 and set(tri_key) == candidate_set
                    for tri_key in self.triangles_metadata.keys()
                )
                if is_triangle_side_pair:
                    shared_vertex = point_names[shared_idx_1]
                    side_end_1 = point_names[end_idx_1]
                    side_end_2 = point_names[end_idx_2]
                break

        if shared_vertex is not None and side_end_1 is not None and side_end_2 is not None:
            shared_pt = self.lookup_pt_by_name(shared_vertex)
            end_pt_1 = self.lookup_pt_by_name(side_end_1)
            end_pt_2 = self.lookup_pt_by_name(side_end_2)

            spread_key = f"equal_distance_triangle_spread_{shared_vertex}_{side_end_1}_{side_end_2}"
            if spread_key not in self.loss_fns:
                min_base_ratio = float(self.opts.get('equal_distance_triangle_min_base_ratio', 0.38))
                spread_weight = float(self.opts.get('equal_distance_triangle_spread_weight', 2200.0))

                self.register_loss(
                    spread_key,
                    lambda s=shared_pt, e1=end_pt_1, e2=end_pt_2, r=min_base_ratio: torch.relu(
                        r * ((self.dist(s, e1) + self.dist(s, e2)) / 2.0) - self.dist(e1, e2)
                    ),
                    weight=spread_weight,
                )

        if self.verbosity:
            logger.info(
                f"Added equal-distance constraint: {point_names[0]}{point_names[1]} = "
                f"{point_names[2]}{point_names[3]} with NDG"
            )

    def _add_fixed_distance_constraint(self, points: list, distance):
        """Add constraint that distance(p1, p2) = fixed_value"""
        if len(points) != 2:
            logger.warning(f"Fixed-distance constraint needs 2 points, got {len(points)}")
            return

        p1 = self.lookup_pt(points[0])
        p2 = self.lookup_pt(points[1])

        # Get the fixed distance value
        target_distance = distance.val if hasattr(distance, 'val') else float(distance)

        self.register_loss(
            f"fixed_distance_{points[0].val}{points[1].val}_{target_distance}",
            lambda pt1=p1, pt2=p2, target=target_distance: self.dist(pt1, pt2) - target,
            weight=50.0  # Giảm xuống - chỉ để constraint nhẹ, không quá quan trọng
        )

        if self.verbosity:
            logger.info(f"Added fixed-distance constraint: {points[0].val}{points[1].val} = {target_distance}")

    def _add_on_circle_constraint(self, points: list):
        if len(points) != 2:
            logger.warning(f"on-circle constraint needs 2 points (point, center), got {len(points)}")
            return

        center_name = points[1].val
        point_name = points[0].val

        # Guard against invalid DSL: triangle vertices are not on the incircle.
        tri_for_incircle = self.defined_incenters.get(center_name)
        if tri_for_incircle and point_name in tri_for_incircle:
            if self.verbosity:
                logger.warning(
                    f"Skipping invalid on-circle constraint: {point_name} on incircle centered at {center_name}"
                )
            return

        point = self.lookup_pt(points[0])
        self.on_circle_pairs.add((point_name, center_name))

        logger.info(f"Adding on-circle constraint: point={points[0].val}, center={center_name}")
        logger.info(f"Available circles: {self.circles}")

        center = self.lookup_pt(points[1])
        center_key = f"center_at_origin_{center_name}"
        if center_key not in self.loss_fns:
            self.register_loss(
                center_key,
                lambda c=center: c.x**2 + c.y**2,
                weight=100.0
            )

        self.register_loss(
            f"on_circle_{points[0].val}_{center_name}",
            lambda pt=point, c=center, cn=center_name: self.dist(pt, c) - self._get_circle_radius(cn),
            weight=100000.0
        )

        # If this is an incircle and the point is explicitly constrained on a side,
        # enforce radius-to-touchpoint perpendicular to that side (true tangency).
        circle_info = None
        for circle_name, info in self.circles:
            if circle_name == center_name:
                circle_info = info
                break

        if (
            circle_info
            and circle_info.get('type') == 'incircle'
            and point_name in self.point_on_segment_defs
        ):
            seg_a_name, seg_b_name = self.point_on_segment_defs[point_name]

            tri = self._get_incircle_point_names(circle_info)
            valid_sides = set()
            if len(tri) >= 3:
                for idx in range(len(tri)):
                    s1 = tri[idx]
                    s2 = tri[(idx + 1) % len(tri)]
                    valid_sides.add(frozenset((s1, s2)))

            if not valid_sides or frozenset((seg_a_name, seg_b_name)) in valid_sides:
                seg_a = self.lookup_pt_by_name(seg_a_name)
                seg_b = self.lookup_pt_by_name(seg_b_name)
                if seg_a is not None and seg_b is not None:
                    tangent_key = f"incircle_tangent_{point_name}_{center_name}_{seg_a_name}{seg_b_name}"
                    if tangent_key not in self.loss_fns:
                        self.register_loss(
                            tangent_key,
                            lambda c=center, p=point, a=seg_a, b=seg_b: self.perpendicular(c, p, a, b),
                            weight=30000.0,
                        )

    def _add_tangent_constraint(self, objects: list):
        """
        Add tangent constraint for line to circle.

        Supported formats:
        1. (tangent A B O) - line AB tangent to circle O (legacy) - 3 objects
        2. (tangent M O A B) - line AB tangent to circle O at point M - 4 objects (from parser)

        Args:
            objects: Can be:
                - [A, B, O] where AB is line and O is circle center (legacy)
                - [M, O, A, B] where M is tangent point, O is center, AB is line (from parser)
        """
        if len(objects) == 4:
            # Format from parser: [tangent_point_M, circle_center_O, line_point_A, line_point_B]
            tangent_point_obj = objects[0]
            center_obj = objects[1]
            p1_obj = objects[2]
            p2_obj = objects[3]

            self._add_line_circle_tangent_with_parsed_points(tangent_point_obj, center_obj, p1_obj, p2_obj)

        elif len(objects) == 3:
            # Check if second object is a circle specification
            obj2 = objects[1]

            if hasattr(obj2, '__iter__') and not isinstance(obj2, str):
                # format [A, (circle_type, O), segment_identifier]
                tangent_point = objects[0]
                circle_center = obj2[1] if len(obj2) > 1 else obj2[0]
                self._add_line_circle_tangent_with_tangent_point(tangent_point, circle_center, objects[2])
            else:
                # Legacy format: (tangent A B O)
                p1 = objects[0]
                p2 = objects[1]
                center_obj = objects[2]
                self._add_line_circle_tangent_by_points(p1, p2, center_obj)
        else:
            logger.warning(f"Tangent constraint needs 3 or 4 objects, got {len(objects)}")

    def _record_tangent_spec(self, tangent_point_name, center_name, p1_name, p2_name):
        """Store unique tangent declarations for deterministic post-solve correction."""
        key = (tangent_point_name, center_name, p1_name, p2_name)
        if key in self._tangent_spec_keys:
            return
        self._tangent_spec_keys.add(key)
        self.tangent_specs.append(
            {
                'tangent_point': tangent_point_name,
                'center': center_name,
                'p1': p1_name,
                'p2': p2_name,
            }
        )

    def _add_line_circle_tangent_by_points(self, p1_obj, p2_obj, center_obj):
        """Add tangent constraint for line (p1, p2) to circle with center"""
        try:
            p1 = self.lookup_pt(p1_obj)
            p2 = self.lookup_pt(p2_obj)
            center = self.lookup_pt(center_obj)
            center_name = center_obj.val

            self.register_loss(
                f"tangent_line_{p1_obj.val}{p2_obj.val}_circle_{center_name}",
                lambda pt1=p1, pt2=p2, c=center, cn=center_name: self.tangent_line_circle(pt1, pt2, c, self._get_circle_radius(cn)),
                weight=20000.0  # Strongly enforce line-circle tangency
            )

            # If one endpoint is intended as tangent point, force it on the circle.
            self.register_loss(
                f"tangent_endpoint_on_circle_{p1_obj.val}_{center_name}",
                lambda pt=p1, c=center, cn=center_name: self.dist(pt, c) - self._get_circle_radius(cn),
                weight=30000.0,
            )

            self._record_tangent_spec(p1_obj.val, center_name, p1_obj.val, p2_obj.val)

            if self.verbosity:
                logger.info(f"Added line-circle tangent: line {p1_obj.val}{p2_obj.val} tangent to circle {center_name}")
        except Exception as e:
            logger.warning(f"Failed to add line-circle tangent: {e}")

    def _add_line_circle_tangent_with_parsed_points(self, tangent_point_obj, center_obj, p1_obj, p2_obj):
        """
        Add tangent constraint from parser format: [M, O, A, B]
        Format: (tangent M (circle O) AB) parsed as 4 separate objects

        Args:
            tangent_point_obj: Point M where line touches circle
            center_obj: Circle center O
            p1_obj: Line point A
            p2_obj: Line point B
        """
        try:
            tangent_point = self.lookup_pt(tangent_point_obj)
            center = self.lookup_pt(center_obj)
            p1 = self.lookup_pt(p1_obj)
            p2 = self.lookup_pt(p2_obj)

            center_name = center_obj.val
            tangent_pt_name = tangent_point_obj.val
            p1_name = p1_obj.val
            p2_name = p2_obj.val


            # CONSTRAINT 1: Khoảng cách từ tâm đến đường thẳng = bán kính
            self.register_loss(
                f"tangent_line_{p1_name}{p2_name}_circle_{center_name}_at_{tangent_pt_name}",
                lambda pt1=p1, pt2=p2, c=center, cn=center_name: self.tangent_line_circle(pt1, pt2, c, self._get_circle_radius(cn)),
                weight=20000.0
            )

            # CONSTRAINT 1.1: Điểm tiếp xúc phải nằm trên đường tròn.
            self.register_loss(
                f"tangent_point_on_circle_{tangent_pt_name}_{center_name}",
                lambda m=tangent_point, c=center, cn=center_name: self.dist(m, c) - self._get_circle_radius(cn),
                weight=30000.0,
            )

            # CONSTRAINT 2: OM vuông góc với AB (tính chất tiếp tuyến)
            self.register_loss(
                f"tangent_perpendicular_{tangent_pt_name}_{center_name}_{p1_name}{p2_name}",
                lambda m=tangent_point, o=center, a=p1, b=p2: self.perpendicular(o, m, a, b)**2,
                weight=20000.0
            )

            # CONSTRAINT 3: M nằm trên đoạn AB (không nằm ngoài)
            self.register_loss(
                f"tangent_point_on_segment_{tangent_pt_name}_{p1_name}{p2_name}",
                lambda m=tangent_point, a=p1, b=p2: self._point_on_segment_loss(m, a, b),
                weight=5000.0  # CỰC KỲ QUAN TRỌNG - M phải nằm trên AB
            )

            # CONSTRAINT 4: Auto-ensure A, B không nằm TRONG đường tròn
            for pt_name, pt in [(p1_name, p1), (p2_name, p2)]:
                if pt_name != tangent_pt_name:
                    # Điểm này phải nằm NGOÀI đường tròn rõ ràng
                    # Nếu chỉ > radius một chút, line vẫn có thể cắt circle
                    # Cần >= radius * 1.3 để chắc chắn line không cắt
                    # Kiểm tra xem constraint đã tồn tại chưa (tránh duplicate khi có nhiều tiếp tuyến)
                    constraint_key = f"outside_circle_{pt_name}_{center_name}"
                    if constraint_key not in self.loss_fns:
                        self.register_loss(
                            constraint_key,
                            lambda c=center, pt=pt, cn=center_name: torch.relu(1.3 * self._get_circle_radius(cn) - self.dist(c, pt))**2,
                            weight=10000.0  # TĂNG CỰC MẠNH - ưu tiên cao hơn để không cắt circle
                        )
                        if self.verbosity:
                            logger.info(f"Auto-added constraint: {pt_name} must stay outside circle {center_name}")
                    else:
                        if self.verbosity:
                            logger.info(f"Constraint {constraint_key} already exists, skipping")

            self._record_tangent_spec(tangent_pt_name, center_name, p1_name, p2_name)

            if self.verbosity:
                logger.info(f"Added line-circle tangent: line {p1_name}{p2_name} tangent to circle {center_name} at {tangent_pt_name}")

        except Exception as e:
            logger.warning(f"Failed to add line-circle tangent with parsed points: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")

    def _add_line_circle_tangent_with_tangent_point(self, tangent_point_obj, center_obj, segment_identifier):
        """
        Add tangent constraint with explicit tangent point.
        Format: (tangent M (circle O) AB)

        Args:
            tangent_point_obj: Point M where line touches circle
            center_obj: Circle center O
            segment_identifier: Segment name or identifier (e.g., "AB" or list of points)
        """
        try:
            tangent_point = self.lookup_pt(tangent_point_obj)
            center = self.lookup_pt(center_obj)
            center_name = center_obj.val


            # Parse segment identifier to get line points
            # segment_identifier could be a string "AB" or a Point object
            if hasattr(segment_identifier, 'val'):
                # It's a point-like object with a name like "AB"
                seg_name = segment_identifier.val
                # Extract point names (assuming format like "AB" means points A and B)
                if len(seg_name) >= 2:
                    p1_name = seg_name[0]
                    p2_name = seg_name[1]

                    # Look up the actual points
                    try:
                        p1 = self.name2pt[p1_name]
                        p2 = self.name2pt[p2_name]
                    except KeyError:
                        logger.warning(f"Could not find points {p1_name} or {p2_name} for segment {seg_name}")
                        return

                    # CONSTRAINT 1: Khoảng cách từ tâm đến đường thẳng = bán kính
                    self.register_loss(
                        f"tangent_line_{p1_name}{p2_name}_circle_{center_name}_at_{tangent_point_obj.val}",
                        lambda pt1=p1, pt2=p2, c=center, cn=center_name: self.tangent_line_circle(pt1, pt2, c, self._get_circle_radius(cn)),
                        weight=20000.0
                    )

                    self.register_loss(
                        f"tangent_point_on_circle_{tangent_point_obj.val}_{center_name}",
                        lambda m=tangent_point, c=center, cn=center_name: self.dist(m, c) - self._get_circle_radius(cn),
                        weight=30000.0,
                    )

                    # CONSTRAINT 2: OM vuông góc với AB (tính chất tiếp tuyến)
                    self.register_loss(
                        f"tangent_perpendicular_{tangent_point_obj.val}_{center_name}_{p1_name}{p2_name}",
                        lambda m=tangent_point, o=center, a=p1, b=p2: self.perpendicular(o, m, a, b)**2,
                        weight=20000.0
                    )

                    # CONSTRAINT 3: M nằm trên đoạn AB (không nằm ngoài)
                    self.register_loss(
                        f"tangent_point_on_segment_{tangent_point_obj.val}_{p1_name}{p2_name}",
                        lambda m=tangent_point, a=p1, b=p2: self._point_on_segment_loss(m, a, b),
                        weight=5000.0  # CỰC KỲ QUAN TRỌNG - M phải nằm trên AB
                    )

                    # CONSTRAINT 4: Auto-ensure A, B không nằm TRONG đường tròn
                    tangent_pt_name = tangent_point_obj.val
                    for pt_name, pt in [(p1_name, p1), (p2_name, p2)]:
                        if pt_name != tangent_pt_name:
                            # Check duplicate constraint trước khi register
                            constraint_key = f"outside_circle_{pt_name}_{center_name}"
                            if constraint_key not in self.loss_fns:
                                self.register_loss(
                                    constraint_key,
                                    lambda c=center, pt=pt, cn=center_name: torch.relu(1.3 * self._get_circle_radius(cn) - self.dist(c, pt))**2,
                                    weight=10000.0  # TĂNG CỰC MẠNH - ưu tiên cao hơn để không cắt circle
                                )
                                if self.verbosity:
                                    logger.info(f"Auto-added constraint: {pt_name} must stay outside circle {center_name}")
                            else:
                                if self.verbosity:
                                    logger.info(f"Constraint {constraint_key} already exists, skipping")

                    self._record_tangent_spec(tangent_point_obj.val, center_name, p1_name, p2_name)

                    if self.verbosity:
                        logger.info(f"Added line-circle tangent with tangent point: line {p1_name}{p2_name} tangent to circle {center_name} at {tangent_point_obj.val}")
                else:
                    logger.warning(f"Segment identifier {seg_name} does not have at least 2 characters")
            else:
                logger.warning(f"Segment identifier is not a point-like object with .val attribute")
        except Exception as e:
            logger.warning(f"Failed to add line-circle tangent with tangent point: {e}")

    def _enforce_chord_length(self):
        """
        Enforce độ dài dây cung không qua tâm để hình vẽ đẹp.
        Tự động phát hiện các dây cung (2 điểm cùng trên 1 circle) không qua tâm
        và enforce độ dài ≈ 0.75-0.8 × đường kính.
        """
        if self.verbosity:
            logger.info("\n=== Enforcing chord lengths ===")
            logger.info(f"Total segments: {len(self.segments)}")
            logger.info(f"Total circles: {len(self.circles)}")

        for seg in self.segments:
            p1_name, p2_name = seg

            # Kiểm tra xem cả 2 điểm có cùng nằm trên 1 đường tròn không
            for circle_name, circle_info in self.circles:
                p1_on_this_circle = any(f"on_circle_{p1_name}_{circle_name}" in key for key in self.loss_fns.keys())
                p2_on_this_circle = any(f"on_circle_{p2_name}_{circle_name}" in key for key in self.loss_fns.keys())

                if p1_on_this_circle and p2_on_this_circle:
                    if self.verbosity:
                        logger.info(f"Segment {p1_name}{p2_name} has both points on circle {circle_name}")

                    # CRITICAL: Kiểm tra xem có phải đường kính không
                    is_diameter = any(
                        f"diameter_midpoint_{circle_name}_{p1_name}_{p2_name}" in key or
                        f"diameter_midpoint_{circle_name}_{p2_name}_{p1_name}" in key
                        for key in self.loss_fns.keys()
                    )

                    if is_diameter:
                        if self.verbosity:
                            logger.info(f"  -> DIAMETER detected! Skipping chord length enforcement.")
                        break  # Skip to next segment

                    # Đây là dây cung KHÔNG qua tâm - enforce độ dài ≈ 0.75 × đường kính
                    radius = circle_info.get('radius')
                    if radius is not None:
                        # Độ dài mục tiêu: 0.75 × đường kính = 1.5 × radius
                        target_length = 1.5 * radius
                        p1 = self.lookup_pt(Parameter(p1_name))
                        p2 = self.lookup_pt(Parameter(p2_name))
                        target_const = self.const(target_length)

                        self.register_loss(
                            f"chord_length_{p1_name}_{p2_name}_{circle_name}",
                            lambda pt1=p1, pt2=p2, target=target_const: (self.dist(pt1, pt2) - target)**2,
                            weight=100.0
                        )

                        if self.verbosity:
                            logger.info(f"  -> Enforcing chord length: {p1_name}{p2_name} ≈ {target_length:.4f} (0.75 × diameter)")
                    break

    def _add_all_chord_ndgs(self):
        """
        Thêm constraint HARD cho TẤT CẢ chords để ngăn đi qua tâm.
        Duyệt qua tất cả cặp (điểm, center) từ on-circle constraints.
        """
        if self.verbosity:
            logger.info(f"\nChecking for chords to add NDG...")
            logger.info(f"Segments: {self.segments}")

        # Thu thập tất cả điểm on-circle theo center
        circle_points = {}  # {center_name: [point_names]}

        for loss_name in self.loss_fns.keys():
            if loss_name.startswith("on_circle_"):
                parts = loss_name.split("_")
                if len(parts) >= 4:  # on_circle_PointName_CenterName
                    point_name = parts[2]
                    center_name = "_".join(parts[3:])

                    if center_name not in circle_points:
                        circle_points[center_name] = []
                    circle_points[center_name].append(point_name)

        if self.verbosity:
            logger.info(f"Circle points: {circle_points}")

        # Với mỗi tâm, kiểm tra các cặp điểm có segment không
        for center_name, point_names in circle_points.items():
            center = self.lookup_pt_by_name(center_name)
            if not center:
                continue

            # Kiểm tra tất cả cặp điểm
            for i in range(len(point_names)):
                for j in range(i + 1, len(point_names)):
                    p1_name = point_names[i]
                    p2_name = point_names[j]

                    # Kiểm tra có segment giữa 2 điểm này không
                    has_segment = (
                        (p1_name, p2_name) in self.segments or
                        (p2_name, p1_name) in self.segments
                    )

                    if has_segment:
                        # Check if this is a DIAMETER before adding NDG
                        is_diameter = any(
                            f"diameter_midpoint_{center_name}_{p1_name}_{p2_name}" in key or
                            f"diameter_midpoint_{center_name}_{p2_name}_{p1_name}" in key
                            for key in self.loss_fns.keys()
                        )

                        if is_diameter:
                            if self.verbosity:
                                logger.info(f"SKIPPING chord NDG for DIAMETER: {p1_name}{p2_name} through {center_name}")
                            continue  # Skip diameter - it MUST go through center!

                        try:
                            pt1 = self.lookup_pt_by_name(p1_name)
                            pt2 = self.lookup_pt_by_name(p2_name)

                            if pt1 and pt2:
                                chord_key = f"chord_not_diameter_{p1_name}_{p2_name}"
                                if chord_key not in self.loss_fns:
                                    # Penalty lớn khi gần thẳng hàng (collinear → 0)
                                    def chord_constraint(p1=pt1, c=center, p2=pt2):
                                        cross = self.collinear(p1, c, p2)
                                        # Penalty cao khi cross gần 0 (thẳng hàng)
                                        return 1.0 / (cross**2 + 0.01)

                                    self.register_loss(
                                        chord_key,
                                        chord_constraint,
                                        weight=50.0
                                    )
                                    if self.verbosity:
                                        logger.info(f"Added HARD chord constraint: {p1_name}-{center_name}-{p2_name} MUST NOT be collinear")
                        except Exception as e:
                            if self.verbosity:
                                logger.warning(f"Failed to add chord constraint: {e}")

    def _enforce_minimum_angle_between_circle_points(self):
        """
        Enforce minimum angle (45 degrees) between points on the same circle.
        This ensures chords are long enough and diagrams look good.
        Also enforce minimum distance to avoid coincident points.
        """
        # Group points by circle
        circle_points_map = {}  # {center_name: [point_objs]}

        for circle_name, circle_info in self.circles:
            circle_points_map[circle_name] = []

        # Find all points with on_circle constraint
        for loss_name in self.loss_fns.keys():  # Changed from self.losses to self.loss_fns.keys()
            if loss_name.startswith("on_circle_"):
                parts = loss_name.split("_")
                if len(parts) >= 4:  # on_circle_PointName_CenterName
                    point_name = parts[2]
                    center_name = "_".join(parts[3:])

                    if center_name in circle_points_map and point_name in self.name2pt:
                        pt_obj = self.name2pt[point_name]
                        circle_points_map[center_name].append((point_name, pt_obj))

        # For each circle with 2+ points, enforce minimum angle
        min_angle_rad = math.radians(30)  # Reduced from 45 to 30 degrees for more flexibility

        for center_name, points_list in circle_points_map.items():
            if len(points_list) < 2:
                continue

            if center_name not in self.name2pt:
                continue

            center = self.name2pt[center_name]

            # For each pair of points on this circle
            for i in range(len(points_list)):
                for j in range(i + 1, len(points_list)):
                    name1, pt1 = points_list[i]
                    name2, pt2 = points_list[j]

                    # Soft minimum distance: only penalize if points are too close
                    # This avoids coincident points without distorting other constraints.
                    dist_key = f"min_dist_circle_{center_name}_{name1}_{name2}"
                    if dist_key not in self.loss_fns:
                        # Get radius for scaling (fallback to 1.0)
                        radius = 1.0
                        for circle_name, circle_info in self.circles:
                            if circle_name == center_name:
                                radius = circle_info.get('radius', 1.0)
                                break
                        min_dist = 0.25 * radius
                        self.register_loss(
                            dist_key,
                            lambda p1=pt1, p2=pt2, md=min_dist: torch.relu(md - self.dist(p1, p2)),
                            weight=200.0
                        )

                    # Use NDG constraint to softly enforce minimum angle
                    # This prevents points from being too close but allows optimization flexibility
                    def angle_ndg(p1=pt1, p2=pt2, c=center, min_cos=math.cos(min_angle_rad)):
                        # Vectors from center to points
                        v1_x = p1.x - c.x
                        v1_y = p1.y - c.y
                        v2_x = p2.x - c.x
                        v2_y = p2.y - c.y

                        # Dot product and magnitudes
                        dot = v1_x * v2_x + v1_y * v2_y
                        mag1 = torch.sqrt(v1_x**2 + v1_y**2 + 1e-8)
                        mag2 = torch.sqrt(v2_x**2 + v2_y**2 + 1e-8)

                        cos_angle = dot / (mag1 * mag2)

                        # Return value that decreases as angle increases (good for NDG)
                        # We want large angles, so return (1 - cos_angle) which is small when angle is small
                        return 1.0 - cos_angle  # Large when angle small, small when angle large

                    # Use register_ndg for soft constraint
                    ndg_key = f"min_angle_{center_name}_{name1}_{name2}"

                    # Check if this constraint already exists
                    if ndg_key not in self.ndgs:
                        self.register_ndg(
                            ndg_key,
                            angle_ndg,
                            weight=50.0
                        )

    def _add_parallel_constraint(self, points: list):
        """Add parallel constraint between two segments"""
        if len(points) != 4:
            logger.warning(f"Parallel constraint needs 4 points (2 points), got {len(points)}")
            return

        p1 = self.lookup_pt(points[0])
        p2 = self.lookup_pt(points[1])
        p3 = self.lookup_pt(points[2])
        p4 = self.lookup_pt(points[3])

        seg1_name = f"{points[0].val}_{points[1].val}"
        seg2_name = f"{points[2].val}_{points[3].val}"
        self.register_loss(f"parallel_{seg1_name}_{seg2_name}",
                          lambda: self.parallel(p1, p2, p3, p4), weight=4000.0)

    def _add_perpendicular_constraint(self, segments):
        """Add perpendicular constraint between two segments"""
        if len(segments) != 4:
            logger.warning(f"Perpendicular constraint needs 4 points (2 segments), got {len(segments)}")
            return

        p1 = self.lookup_pt(segments[0])
        p2 = self.lookup_pt(segments[1])
        p3 = self.lookup_pt(segments[2])
        p4 = self.lookup_pt(segments[3])

        seg1_name = f"{segments[0].val}_{segments[1].val}"
        seg2_name = f"{segments[2].val}_{segments[3].val}"
        self.register_loss(f"perpendicular_{seg1_name}_{seg2_name}",
                          lambda: self.perpendicular(p1, p2, p3, p4), weight=5000.0)

        # Lưu thông tin perpendicular để renderer vẽ dấu vuông góc
        self.perpendiculars.append((segments[0].val, segments[1].val, segments[2].val, segments[3].val))

    def _add_angle_equal_constraint(self, points: list):
        """Add angle equality constraint: angle ABC = angle DEF"""
        if len(points) != 6:
            logger.warning(f"Angle-equal constraint needs 6 points, got {len(points)}")
            return

        p1 = self.lookup_pt(points[0])
        p2 = self.lookup_pt(points[1])
        p3 = self.lookup_pt(points[2])
        p4 = self.lookup_pt(points[3])
        p5 = self.lookup_pt(points[4])
        p6 = self.lookup_pt(points[5])

        angle1_name = f"{points[0].val}_{points[1].val}_{points[2].val}"
        angle2_name = f"{points[3].val}_{points[4].val}_{points[5].val}"
        self.register_loss(
            f"angle_equal_{angle1_name}_{angle2_name}",
            lambda: self._angle_diff_loss(p1, p2, p3, p4, p5, p6),
            weight=1000.0
        )

        self.angle_equal_assertions.append({
            'angle1': (points[0].val, points[1].val, points[2].val),  # (B, A, D)
            'angle2': (points[3].val, points[4].val, points[5].val)   # (C, A, D)
        })


    def preprocess(self):
        if self.verbosity:
            logger.info("Processing instructions")

        for instr in self.instructions:
            if self.verbosity:
                logger.info(f" {instr}")
            self.process_instruction(instr)

    def train(self, epochs: int = 1000, lr: float = 0.01):
        if not self.has_loss:
            return 0.0

        optimizer = optim.Adam(self.trainable_vars, lr=lr)
        grad_clip = self.opts.get('grad_clip_norm', 5.0)
        param_abs_max = self.opts.get('param_abs_max', 1e3)
        last_good_state = [p.detach().clone() for p in self.trainable_vars]
        total_loss = torch.tensor(float('inf'), dtype=torch.float64, device=self.device)
        non_finite_penalty = self.const(1e6)
        if self.verbosity:
            logger.info(f"Optimization ({epochs}) Epochs")

        for i in range(epochs):
            optimizer.zero_grad()
            # Compute losses fresh at each iteration
            raw_losses = {key: fn() for key, fn in self.loss_fns.items()}
            had_non_finite = False
            self.losses = {}
            for key, value in raw_losses.items():
                if not torch.isfinite(value):
                    had_non_finite = True
                self.losses[key] = torch.nan_to_num(
                    value,
                    nan=non_finite_penalty.item(),
                    posinf=non_finite_penalty.item(),
                    neginf=non_finite_penalty.item(),
                )

            total_loss = sum(self.losses.values())
            if had_non_finite and self.verbosity:
                logger.warning(f"Non-finite term(s) detected at iteration {i}; replaced with finite penalties")

            total_loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.trainable_vars, grad_clip)

            optimizer.step()

            # Keep parameters in a numerically safe range.
            if param_abs_max is not None and param_abs_max > 0:
                for param in self.trainable_vars:
                    param.data.clamp_(-param_abs_max, param_abs_max)

            if torch.isfinite(total_loss):
                last_good_state = [p.detach().clone() for p in self.trainable_vars]

            if self.verbosity and i % 100 == 0:
                logger.info(f"Iteration {i:4d}: Loss = {total_loss.item():.6f}")

            # Early stopping
            if total_loss.item() < 1e-6:
                if self.verbosity >= 0:
                    logger.info(f"Converged at iteration {i} with loss {total_loss.item():.6f}")
                break

        final_loss = total_loss.item()
        if self.verbosity:
            logger.info(f"Final loss {final_loss:.6f}")
            self.log_losses()
        return final_loss

    def log_losses(self):
        if len(self.loss_fns) == 0:
            return
        # Recompute losses for logging
        if not self.losses:
            self.losses = {key: fn() for key, fn in self.loss_fns.items()}
        logger.info("\n Loss breakdown")
        for key, loss in self.losses.items():
            logger.info(f"{key:30s}: {loss.item():.6f}")

    def regularize_points(self):
        """Add regularization to keep points near origin"""
        if len(self.name2pt) > 0:
            def compute_reg():
                norms = [self.norm(p) for p in self.name2pt.values()]
                return torch.stack(norms).mean()
            self.register_loss("regularization", compute_reg, weight=0.05)

    def make_points_distinct(self):
        pts = list(self.name2pt.values())
        if len(pts) < 2:
            return

        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                # Encourage d > 0.1
                self.register_ndg(f"distinct_{i}_{j}",
                                 lambda pi=pts[i], pj=pts[j]: self.dist(pi, pj), weight=0.1)

    def solve_single(self, attempt_id=0):
        self.current_attempt = attempt_id
        self._init_state()
        self.preprocess()

        # Optional anti-diameter NDG for chords; disabled by default because it can over-constrain circle problems.
        if self.opts.get('enable_chord_ndg', False):
            self._add_all_chord_ndgs()
        # self._enforce_chord_length()  # TẠM THỜI TẮT để test

        self.regularize_points()  # Giữ các điểm gần gốc tọa độ
        # self.make_points_distinct()

        loss = float('inf')
        if self.has_loss:
            loss = self.train(epochs=self.opts.get('epochs', 1000),
                      lr=self.opts.get('learning_rate', 0.01))

        return self.get_diagram(), loss

    def solve(self, n_tries=None):
        """Solve with multiple initialization attempts"""
        if n_tries is None:
            n_tries = self.opts.get('n_tries', 1)

        eps = self.opts.get('eps', 1e-6)
        best_loss = float('inf')
        best_diagram = None

        for attempt in range(n_tries):
            if attempt > 0:
                # Reset state for new attempt
                self._init_state()
                # Set different random seed for varied initialization
                random.seed(self.opts.get('seed', 42) + attempt)
                torch.manual_seed(self.opts.get('seed', 42) + attempt)

            if self.verbosity and n_tries > 1:
                logger.info(f"\nAttempt {attempt + 1}/{n_tries}")
            try:
                diagram, loss = self.solve_single(attempt_id=attempt)

                # Early stopping if converged
                if loss < eps:
                    if self.verbosity and n_tries > 1:
                        logger.success(f"Converged at attempt {attempt + 1} with loss {loss:.6f}")
                    return diagram

                # Track best result
                if loss < best_loss:
                    best_loss = loss
                    best_diagram = diagram
                    if self.verbosity and n_tries > 1:
                        logger.info(f"New best loss: {loss:.6f}")

            except Exception as e:
                if self.verbosity:
                    logger.error(f"Attempt {attempt + 1} failed:")
                    logger.exception(e)
                continue

        if self.verbosity and n_tries > 1:
            logger.info(f"\nBest loss after {n_tries} attempts: {best_loss:.6f}")

        return best_diagram if best_diagram is not None else self.get_diagram()

    def get_diagram(self):
        diagram = Diagram()

        # BƯỚC 1: Tính centroid của tất cả điểm
        if len(self.name2pt) > 0:
            all_x = [pt.x.detach().cpu().item() for pt in self.name2pt.values()]
            all_y = [pt.y.detach().cpu().item() for pt in self.name2pt.values()]
            centroid_x = sum(all_x) / len(all_x)
            centroid_y = sum(all_y) / len(all_y)
        else:
            centroid_x = 0.0
            centroid_y = 0.0

        # BƯỚC 2: Dịch chuyển tất cả điểm về (0, 0) - CỐ ĐỊNH CHÍNH GIỮA
        for name, pt in self.name2pt.items():
            x = pt.x.detach().cpu().item() - centroid_x  # Dịch về (0, 0)
            y = pt.y.detach().cpu().item() - centroid_y
            geo_pt = GeometricPoint(x, y, name)
            diagram.add_point(name, geo_pt)

        self._apply_incircle_post_correction(diagram)
        self._apply_tangent_post_correction(diagram)

        # Add triangles with metadata
        for key, metadata in self.triangles_metadata.items():
            p1_name, p2_name, p3_name = key
            if p1_name in diagram.points and p2_name in diagram.points and p3_name in diagram.points:
                p1 = diagram.points[p1_name]
                p2 = diagram.points[p2_name]
                p3 = diagram.points[p3_name]

                equal_sides = metadata.get('equal_sides')
                right_angle_at = metadata.get('right_angle_at')
                equal_angles = metadata.get('equal_angles')

                logger.info(f"Adding triangle {key} with equal_angles: {equal_angles}")
                diagram.add_triangle(p1, p2, p3, equal_sides, right_angle_at, equal_angles)

        # Add quadrilaterals with metadata
        for key, metadata in self.quadrilaterals_metadata.items():
            p1_name, p2_name, p3_name, p4_name = key
            if all(name in diagram.points for name in [p1_name, p2_name, p3_name, p4_name]):
                diagram.add_quadrilateral(
                    diagram.points[p1_name],
                    diagram.points[p2_name],
                    diagram.points[p3_name],
                    diagram.points[p4_name],
                    metadata
                )

        # Add circles
        for center_name, info in self.circles:
            if center_name in diagram.points:
                center = diagram.points[center_name]
                center_pt = self.name2pt[center_name]

                # Calculate radius based on circle type
                if info['type'] == 'incircle':
                    # Radius = distance from center to any boundary side
                    boundary_points = self._get_incircle_point_names(info)
                    if len(boundary_points) < 2:
                        continue
                    p1 = diagram.points.get(boundary_points[0])
                    p2 = diagram.points.get(boundary_points[1])
                    if p1 is not None and p2 is not None:
                        radius = self._distance_point_to_line(center, p1, p2)
                    else:
                        p1_t = self.name2pt[boundary_points[0]]
                        p2_t = self.name2pt[boundary_points[1]]
                        radius = self.dist_to_line(center_pt, p1_t, p2_t).detach().cpu().item()
                    info = {**info, 'radius': radius}

                elif info['type'] == 'circumcircle':
                    # Radius = distance from center to any circumcircle reference point
                    circum_points = self._get_circumcircle_point_names(info)
                    if len(circum_points) < 1:
                        continue
                    p1 = diagram.points.get(circum_points[0])
                    if p1 is not None:
                        radius = math.hypot(center.x - p1.x, center.y - p1.y)
                    else:
                        p1_t = self.name2pt[circum_points[0]]
                        radius = self.dist(center_pt, p1_t).detach().cpu().item()
                    info = {**info, 'radius': radius}
                elif info['type'] == 'diameter':
                    endpoints = info.get('endpoints', [])
                    if endpoints and endpoints[0] in diagram.points:
                        p1 = diagram.points[endpoints[0]]
                        radius = math.hypot(center.x - p1.x, center.y - p1.y)
                    else:
                        p1 = self.lookup_pt_by_name(endpoints[0]) if endpoints else None
                        if p1 is not None:
                            radius = self.dist(center_pt, p1).detach().cpu().item()
                        else:
                            radius = 1.0
                    info = {**info, 'radius': radius}
                # For 'positioned' type, radius is already in info

                diagram.add_circle(center, info)

        for p1_name, p2_name in self.segments:
            if p1_name in diagram.points and p2_name in diagram.points:
                p1 = diagram.points[p1_name]
                p2 = diagram.points[p2_name]
                diagram.add_segment(p1, p2)

        for p1_name, p2_name in self.lines:
            if p1_name in diagram.points and p2_name in diagram.points:
                p1 = diagram.points[p1_name]
                p2 = diagram.points[p2_name]
                # Store line as tuple for rendering
                line_name = f"line_{p1_name}_{p2_name}"
                diagram.add_line(line_name, (p1, p2))

        # Add angle bisectors
        if hasattr(self, 'angle_bisectors_metadata'):
            for bisector_data in self.angle_bisectors_metadata:
                # Convert point names to GeometricPoint objects
                vertex_name = bisector_data['vertex']
                bisector_point_name = bisector_data['bisector_point']

                if vertex_name in diagram.points and bisector_point_name in diagram.points:
                    diagram.angle_bisectors.append({
                        'vertex': diagram.points[vertex_name],
                        'point': diagram.points[bisector_point_name],
                        'angle_points': bisector_data.get('angle_points', [])
                    })

        # Add angle-equal assertions
        for assertion in self.angle_equal_assertions:
            angle1 = assertion['angle1']  # (p1, p2, p3)
            angle2 = assertion['angle2']  # (p4, p5, p6)

            if all(pname in diagram.points for pname in angle1 + angle2):
                diagram.angle_equal_assertions.append({
                    'angle1': {
                        'p1': diagram.points[angle1[0]],
                        'vertex': diagram.points[angle1[1]],
                        'p2': diagram.points[angle1[2]]
                    },
                    'angle2': {
                        'p1': diagram.points[angle2[0]],
                        'vertex': diagram.points[angle2[1]],
                        'p2': diagram.points[angle2[2]]
                    }
                })

        # Add angle measures for display
        for vertex_name, p1_name, p2_name, degrees in self.angle_measures:
            if all(pname in diagram.points for pname in [vertex_name, p1_name, p2_name]):
                diagram.add_angle_measure(
                    diagram.points[vertex_name],
                    diagram.points[p1_name],
                    diagram.points[p2_name],
                    degrees
                )

        # Add perpendicular constraints for rendering perpendicular markers
        for p1_name, p2_name, p3_name, p4_name in self.perpendiculars:
            if all(pname in diagram.points for pname in [p1_name, p2_name, p3_name, p4_name]):
                diagram.perpendiculars.append((
                    diagram.points[p1_name],
                    diagram.points[p2_name],
                    diagram.points[p3_name],
                    diagram.points[p4_name]
                ))

        return diagram

    def _get_circle_radius_by_center(self, center_name: str):
        """Resolve circle radius by center name from parsed circle metadata."""
        for circle_center_name, info in self.circles:
            if circle_center_name != center_name:
                continue

            circle_type = info.get('type')
            if circle_type == 'positioned':
                radius = info.get('radius')
                return float(radius) if radius is not None else None

            if circle_type == 'incircle':
                boundary_points = self._get_incircle_point_names(info)
                if len(boundary_points) < 2:
                    return None
                center_pt = self.lookup_pt_by_name(center_name)
                p1 = self.lookup_pt_by_name(boundary_points[0])
                p2 = self.lookup_pt_by_name(boundary_points[1])
                if center_pt is None or p1 is None or p2 is None:
                    return None
                return float(self.dist_to_line(center_pt, p1, p2).detach().cpu().item())

            if circle_type == 'circumcircle':
                circum_points = self._get_circumcircle_point_names(info)
                if len(circum_points) < 1:
                    return None
                center_pt = self.lookup_pt_by_name(center_name)
                p1 = self.lookup_pt_by_name(circum_points[0])
                if center_pt is None or p1 is None:
                    return None
                return float(self.dist(center_pt, p1).detach().cpu().item())

            radius = info.get('radius')
            return float(radius) if radius is not None else None

        return None

    def _apply_tangent_post_correction(self, diagram: Diagram):
        """Deterministically project tangent geometry so rendered tangent does not cut circle."""
        for spec in self.tangent_specs:
            center_name = spec['center']
            p1_name = spec['p1']
            p2_name = spec['p2']
            tangent_name = spec['tangent_point']

            if center_name not in diagram.points:
                continue
            if p1_name not in diagram.points or p2_name not in diagram.points:
                continue
            if tangent_name not in diagram.points:
                continue

            radius = self._get_circle_radius_by_center(center_name)
            if radius is None or radius <= 1e-8:
                continue

            center_pt = diagram.points[center_name]
            tangent_pt = diagram.points[tangent_name]
            p1 = diagram.points[p1_name]
            p2 = diagram.points[p2_name]

            vx = tangent_pt.x - center_pt.x
            vy = tangent_pt.y - center_pt.y
            norm_v = math.hypot(vx, vy)
            if norm_v < 1e-8:
                fallback = p1 if tangent_name != p1_name else p2
                vx = fallback.x - center_pt.x
                vy = fallback.y - center_pt.y
                norm_v = math.hypot(vx, vy)
            if norm_v < 1e-8:
                vx, vy = 1.0, 0.0
                norm_v = 1.0

            ux = vx / norm_v
            uy = vy / norm_v

            # Move tangent point exactly onto circle.
            tangent_pt.x = center_pt.x + radius * ux
            tangent_pt.y = center_pt.y + radius * uy

            # Tangent direction is perpendicular to radius OM.
            tx = -uy
            ty = ux

            if tangent_name in (p1_name, p2_name):
                other = p2 if tangent_name == p1_name else p1
                sx = other.x - tangent_pt.x
                sy = other.y - tangent_pt.y
                proj = sx * tx + sy * ty
                if abs(proj) < radius * 0.8:
                    proj = radius * 1.2 if proj >= 0 else -radius * 1.2
                other.x = tangent_pt.x + proj * tx
                other.y = tangent_pt.y + proj * ty
                continue

            # If tangent point is separate from line endpoints, project both endpoints onto tangent line.
            s1 = (p1.x - tangent_pt.x) * tx + (p1.y - tangent_pt.y) * ty
            s2 = (p2.x - tangent_pt.x) * tx + (p2.y - tangent_pt.y) * ty

            if abs(s1) < radius * 0.6 and abs(s2) < radius * 0.6:
                s1 = -radius * 1.1
                s2 = radius * 1.1
            elif abs(s1) < radius * 0.6:
                s1 = -radius * 1.1 if s2 >= 0 else radius * 1.1
            elif abs(s2) < radius * 0.6:
                s2 = radius * 1.1 if s1 >= 0 else -radius * 1.1

            p1.x = tangent_pt.x + s1 * tx
            p1.y = tangent_pt.y + s1 * ty
            p2.x = tangent_pt.x + s2 * tx
            p2.y = tangent_pt.y + s2 * ty

    def _distance_point_to_line(self, point, p1, p2):
        """Euclidean distance from point to the infinite line through p1-p2 (diagram coords)."""
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        denom = math.hypot(dx, dy)
        if denom < 1e-12:
            return 0.0
        num = abs(dy * point.x - dx * point.y + p2.x * p1.y - p2.y * p1.x)
        return num / denom

    def _project_point_to_line(self, point, p1, p2):
        """Orthogonal projection of point onto line p1-p2 in diagram coordinates."""
        vx = p2.x - p1.x
        vy = p2.y - p1.y
        vv = vx * vx + vy * vy
        if vv < 1e-12:
            return (p1.x, p1.y)
        t = ((point.x - p1.x) * vx + (point.y - p1.y) * vy) / vv
        return (p1.x + t * vx, p1.y + t * vy)

    def _compute_incenter_coords(self, a, b, c):
        """Compute incenter from triangle vertices in diagram coordinates."""
        side_a = math.hypot(b.x - c.x, b.y - c.y)  # opposite A
        side_b = math.hypot(c.x - a.x, c.y - a.y)  # opposite B
        side_c = math.hypot(a.x - b.x, a.y - b.y)  # opposite C
        total = side_a + side_b + side_c
        if total < 1e-12:
            return None
        x = (side_a * a.x + side_b * b.x + side_c * c.x) / total
        y = (side_a * a.y + side_b * b.y + side_c * c.y) / total
        return (x, y)

    def _apply_incircle_post_correction(self, diagram: Diagram):
        """Force incircle center/touchpoints to be geometrically consistent for rendering."""
        for center_name, info in self.circles:
            if info.get('type') != 'incircle':
                continue
            boundary_points = self._get_incircle_point_names(info)
            if len(boundary_points) < 3:
                continue
            if center_name not in diagram.points:
                continue
            if not all(name in diagram.points for name in boundary_points):
                continue

            center = diagram.points[center_name]

            # Keep exact analytical center only for triangle incircle.
            if len(boundary_points) == 3:
                a = diagram.points[boundary_points[0]]
                b = diagram.points[boundary_points[1]]
                c = diagram.points[boundary_points[2]]

                incenter_xy = self._compute_incenter_coords(a, b, c)
                if incenter_xy is None:
                    continue

                center.x, center.y = incenter_xy

            side_to_touch = {}
            for point_name, seg in self.point_on_segment_defs.items():
                if (point_name, center_name) not in self.on_circle_pairs:
                    continue
                side_to_touch[frozenset(seg)] = point_name

            polygon_sides = []
            for idx in range(len(boundary_points)):
                polygon_sides.append((boundary_points[idx], boundary_points[(idx + 1) % len(boundary_points)]))

            for s1_name, s2_name in polygon_sides:
                point_name = side_to_touch.get(frozenset((s1_name, s2_name)))
                if not point_name or point_name not in diagram.points:
                    continue
                s1 = diagram.points[s1_name]
                s2 = diagram.points[s2_name]
                tx, ty = self._project_point_to_line(center, s1, s2)
                touch = diagram.points[point_name]
                touch.x = tx
                touch.y = ty
