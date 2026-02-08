from networkx import center
import torch
import torch.nn as nn
import torch.optim as optim
import random
from loguru import logger
from collections import namedtuple
from src.models.domain.geometry.instructions import Parameter, Assertion
from src.models.domain.geometry.value_objects import Point, Line
from src.models.domain.geometry.entities import GeometricPoint, Diagram
from src.models.domain.geometry.types import QuadrilateralType, TriangleType, DiagramType
from src.services.diagram.initializer import Initializer

from loguru import logger

TorchPoint = namedtuple("TorchPoint", ["x", "y"])
LineSF = namedtuple("LineSF", ["a", "b", "c", "p1", "p2"])
LineNF = namedtuple("LineNF", ["n", "f"])

class Optimizer:
    def __init__(self, instructions, epochs=1000, n_tries=1, learning_rate=0.01, eps=1e-6, seed=42, verbosity=False):
        self.instructions = instructions
        self.epochs = epochs
        self.n_tries = n_tries
        self.learning_rate = learning_rate
        self.eps = eps
        self.seed = seed
        self.verbosity = verbosity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_state()  # Initialize all state variables

    class CircleAngleConstraint:
        """Reusable constraint for enforcing minimum angle between points on a circle."""
        
        def __init__(self, optimizer, pt1: TorchPoint, pt2: TorchPoint, center: TorchPoint, min_cos_value):
            self.optimizer = optimizer
            self.pt1 = pt1
            self.pt2 = pt2
            self.center = center
            self.min_cos_value = min_cos_value
        
        def __call__(self):
            """Compute constraint loss: penalty if angle < threshold."""
            cos_angle = self.optimizer.angle_cosine(self.pt1, self.center, self.pt2)
            diff = cos_angle - self.min_cos_value
            return torch.where(diff > 0, diff, torch.zeros_like(diff))**2

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
        return torch.sqrt(dx**2 + dy**2)

    def norm(self, p: TorchPoint):
        return torch.sqrt(p.x**2 + p.y**2)

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
        n_norm = torch.sqrt(n_x**2 + n_y**2)
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
            assert p.val not in self.name2pt, f"Point {p.val} already registered"
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
        
        quad_type = quad_type.lower()
        
        # Initialize coordinates based on quad type
        if quad_type == 'square':
            init_coords = Initializer.init_square(1.0)
        elif quad_type == 'rectangle':
            init_coords = Initializer.init_rectangle(1.6, 1.0)
        elif quad_type == 'parallelogram':
            init_coords = Initializer.init_parallelogram(1.0)
        elif quad_type == 'trapezoid':
            init_coords = Initializer.init_trapezoid(1.0)
        elif quad_type == 'rhombus':
            init_coords = Initializer.init_rhombus(1.0)
        else:  # generic quadrilateral
            init_coords = Initializer.init_quadrilateral(1.0)
        
        # Add noise to avoid perfect initialization
        init_coords = Initializer.add_noise(init_coords, noise_scale=0.05)
        
        # Create points with initial coordinates
        pt_objs = [self.sample_uniform(p, init_coords=init_coords[i]) for i, p in enumerate(points)]
        names = [p.val for p in points]
        self.quadrilaterals.append(tuple(names))
        
        p1, p2, p3, p4 = pt_objs
        key = tuple(names)
        
        # Apply constraints based on quad type
        if quad_type == 'square':
            # All sides equal + right angle
            self.register_loss(f"sq_eq_12_23_{names[0]}", lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=10.0)
            self.register_loss(f"sq_eq_23_34_{names[0]}", lambda: self.dist(p2, p3) - self.dist(p3, p4), weight=10.0)
            self.register_loss(f"sq_eq_34_41_{names[0]}", lambda: self.dist(p3, p4) - self.dist(p4, p1), weight=10.0)
            self.register_loss(f"sq_right_B_{names[0]}", lambda: self._dot_product(p1, p2, p3), weight=10.0)
            self.register_ndg(f"sq_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            
            self.quadrilaterals_metadata[key] = {
                'type': QuadrilateralType.SQUARE,
                'equal_sides': [(0, 1), (1, 2), (2, 3), (3, 0)]
            }
            
        elif quad_type == 'rectangle':
            # Three right angles
            self.register_loss(f"rect_right_B_{names[0]}", lambda: self._dot_product(p1, p2, p3), weight=10.0)
            self.register_loss(f"rect_right_C_{names[0]}", lambda: self._dot_product(p2, p3, p4), weight=10.0)
            self.register_loss(f"rect_right_D_{names[0]}", lambda: self._dot_product(p3, p4, p1), weight=10.0)
            self.register_ndg(f"rect_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            
            self.quadrilaterals_metadata[key] = {
                'type': QuadrilateralType.RECTANGLE,
                'equal_sides': [(0, 2), (1, 3)]  # Opposite sides equal
            }
            
        elif quad_type == 'parallelogram':
            # Opposite sides parallel
            self.register_loss(f"para_parallel_12_34_{names[0]}", 
                              lambda: self.parallel(p1, p2, p4, p3), weight=10.0)
            self.register_loss(f"para_parallel_23_41_{names[0]}", 
                              lambda: self.parallel(p2, p3, p1, p4), weight=10.0)
            # Opposite sides equal (helps convergence)
            self.register_loss(f"para_eq_12_34_{names[0]}", lambda: self.dist(p1, p2) - self.dist(p4, p3), weight=5.0)
            self.register_loss(f"para_eq_23_41_{names[0]}", lambda: self.dist(p2, p3) - self.dist(p1, p4), weight=5.0)
            self.register_ndg(f"para_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            
            self.quadrilaterals_metadata[key] = {'type': 'parallelogram', 'opposite_parallel': True}
            
        elif quad_type == 'trapezoid':
            # One pair parallel: AB || CD
            self.register_loss(f"trap_parallel_12_43_{names[0]}",
                              lambda: self.parallel(p1, p2, p4, p3), weight=10.0)
            self.register_ndg(f"trap_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            # Ensure bases have reasonable lengths
            self.register_ndg(f"trap_ndg_base1_{names[0]}", lambda: self.dist(p1, p2), weight=10.0)
            self.register_ndg(f"trap_ndg_base2_{names[0]}", lambda: self.dist(p3, p4), weight=10.0)
            
            self.quadrilaterals_metadata[key] = {'type': 'trapezoid', 'parallel_sides': [(0, 1), (3, 2)]}
            
        elif quad_type == 'rhombus':
            # All sides equal
            self.register_loss(f"rhombus_eq_12_23_{names[0]}", lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=10.0)
            self.register_loss(f"rhombus_eq_23_34_{names[0]}", lambda: self.dist(p2, p3) - self.dist(p3, p4), weight=10.0)
            self.register_loss(f"rhombus_eq_34_41_{names[0]}", lambda: self.dist(p3, p4) - self.dist(p4, p1), weight=10.0)
            # Diagonals perpendicular: AC ⊥ BD
            self.register_loss(f"rhombus_diag_perp_{names[0]}",
                              lambda: self.perpendicular(p1, p3, p2, p4), weight=10.0)
            self.register_ndg(f"rhombus_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            # Ensure diagonals have reasonable lengths
            self.register_ndg(f"rhombus_diag_ac_{names[0]}", lambda: self.dist(p1, p3), weight=10.0)
            self.register_ndg(f"rhombus_diag_bd_{names[0]}", lambda: self.dist(p2, p4), weight=10.0)
            
            self.quadrilaterals_metadata[key] = {'type': 'rhombus', 'equal_sides': [(0, 1), (1, 2), (2, 3), (3, 0)]}
            
        else:  # generic quadrilateral
            # Only non-degeneracy constraint
            self.register_ndg(f"quad_area_{names[0]}", lambda: self._cross_product_area(p1, p2, p3), weight=20.0)
            self.quadrilaterals_metadata[key] = {'type': 'quadrilateral'}
        
        return pt_objs


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

        init_coords = Initializer.add_noise(init_coords)

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
                              weight=10.0)
            others = [i for i in range(3) if i != apex_idx]
            metadata['equal_sides'] = [(apex_idx, others[0]), (apex_idx, others[1])]

        if tri_type == 'right' or tri_type == 'right_isosceles':
            right_pt = pts[right_idx]
            other_pts = [pts[i] for i in range(3) if i != right_idx]
            self.register_loss(f"right_{points[0].val}_{points[1].val}_{points[2].val}",
                              lambda: self._dot_product(other_pts[0], right_pt, other_pts[1]), weight=10.0)
            metadata['right_angle_at'] = right_idx

        if tri_type == 'equilateral':
            self.register_loss(f"equi_12_23_{points[0].val}",
                              lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=10.0)
            self.register_loss(f"equi_23_31_{points[0].val}",
                              lambda: self.dist(p2, p3) - self.dist(p3, p1), weight=10.0)
            metadata['equal_sides'] = [(0, 1), (1, 2), (2, 0)]

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
                         lambda: self.collinear(p1, p2, p3), weight=1.0)
        key = (points[0].val, points[1].val, points[2].val)
        self.triangles_metadata[key] = metadata
        return [p1, p2, p3]

    def _define_projection(self, point_name, vertex_point, segment_points):
        # segment_points có thể là [(segment A B)] hoặc [A, B]
        # Nếu là nested segment, extract points từ đó
        if len(segment_points) == 1 and hasattr(segment_points[0], 'objects'):
            # Nested segment: (segment A B)
            actual_points = segment_points[0].objects
        else:
            # Direct points: A B
            actual_points = segment_points
        
        assert len(actual_points) == 2

        foot = self.sample_uniform(point_name)
        vertex = self.lookup_pt(vertex_point)
        p1 = self.lookup_pt(actual_points[0])
        p2 = self.lookup_pt(actual_points[1])

        # Foot perpendicular to segment and lies ON segment (between endpoints)
        self.register_loss(f"perp_{point_name.val}",
                          lambda: self.perpendicular(vertex, foot, p1, p2), weight=10.0)
        self.register_loss(f"on_seg_{point_name.val}",
                          lambda: self.collinear(foot, p1, p2), weight=10.0)
        
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
        center_coords = Initializer.init_circle_with_positioned_points(radius_value, len(points_info))
        center_coords = Initializer.add_noise(center_coords)
        center = self.sample_uniform(center_name, init_coords=center_coords)
        
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
    
    def _define_incenter(self, point_name, triangle_points):
        """Define incenter - equal distance to all sides"""
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        init_coords = Initializer.init_triangle_incircle()
        init_coords = Initializer.add_noise(init_coords)
        incenter = self.sample_uniform(point_name, init_coords=init_coords[3])
        self.register_loss(f"incenter_{point_name.val}",
                          lambda: (self.dist_to_line(incenter, p1, p2) - self.dist_to_line(incenter, p2, p3))**2 +
                                  (self.dist_to_line(incenter, p2, p3) - self.dist_to_line(incenter, p3, p1))**2,
                          weight=10.0)
        return incenter

    def _define_circumcenter(self, point_name, triangle_points):
        """Define circumcenter - equal distance to all vertices"""
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        init_coords = Initializer.init_triangle_circumcircle(radius=1.0)
        init_coords = Initializer.add_noise(init_coords, noise_scale=0.02)
        circumcenter = self.sample_uniform(point_name, init_coords=init_coords[3])

        self.register_loss(f"circumcenter_{point_name.val}",
                          lambda: (self.dist(circumcenter, p1) - self.dist(circumcenter, p2))**2 +
                                  (self.dist(circumcenter, p2) - self.dist(circumcenter, p3))**2,
                          weight=10.0)
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
        midpoint = self.sample_uniform(point_name)
        
        # Constraint 1: Midpoint position (x,y coordinates)
        self.register_loss(f"midpoint_pos_{point_name.val}",
                          lambda: (midpoint.x - (p1.x + p2.x)/2)**2 + (midpoint.y - (p1.y + p2.y)/2)**2,
                          weight=150.0)
        
        # Constraint 2: Collinearity 
        self.register_loss(f"midpoint_collinear_{point_name.val}",
                          lambda: self.collinear(midpoint, p1, p2),
                          weight=100.0)
        
        return midpoint

    def parameter_on_seg(self, p, segment_points: list):
        """Define a point on segment - ensures point is between segment endpoints"""
        assert len(segment_points) == 2
        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])
        
        # Smart initialization: place point ON the segment at random position
        # Use t in range [0.3, 0.7] to avoid being too close to endpoints
        import random
        t = random.uniform(0.3, 0.7)
        init_x = p1.x.item() * (1 - t) + p2.x.item() * t
        init_y = p1.y.item() * (1 - t) + p2.y.item() * t
        
        P = self.sample_uniform(p, init_coords=(init_x, init_y))
        self.register_loss(f"on_seg_{p.val}",
                          lambda pt=P, pt1=p1, pt2=p2: self._point_on_segment_loss(pt, pt1, pt2),
                          weight=50.0)
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
        point = self.sample_uniform(point_name)
        self.register_loss(f"perp_bisector_{point_name.val}",
                          lambda: (self.dist(point, p1) - self.dist(point, p2))**2,
                          weight=10.0)
        return point

    def process_instruction(self, instr):
        from src.models.domain.geometry.instructions import Assertion
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

        # Constraint 1: D nằm trên đoạn BC
        self.register_loss(
            f"bisector_on_segment_{key}",
            lambda: self._point_on_segment_loss(p_bisector, p1, p2),
            weight=10.0
        )

        # Constraint 2: Góc BAD = góc CAD (tính chất phân giác)
        self.register_loss(
            f"bisector_equal_angle_{key}",
            lambda: self._angle_bisector_equal_loss(p_vertex, p1, p2, p_bisector),
            weight=10.0
        )

        # Constraint 3: BD/DC = AB/AC (định lý phân giác)
        self.register_loss(
            f"bisector_ratio_{key}",
            lambda: self._angle_bisector_ratio_loss(p_vertex, p1, p2, p_bisector),
            weight=5.0
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

        # Valid quadrilateral types
        valid_types = {"square", "rectangle", "parallelogram", "trapezoid", "rhombus", "quadrilateral"}
        
        # Use the param_type_str if valid, otherwise fallback to generic
        quad_type = param_type_str if param_type_str in valid_types else "quadrilateral"
        
        # Call the unified sample_quadrilateral function
        self.sample_quadrilateral(objects, quad_type=quad_type)
        
        logger.info(f"Processed quadrilateral type: {quad_type}")

    def _process_point_parameter(self, param_type, objects, args):
        param_type_str = str(param_type).lower() if param_type else ""

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
        
        # CỐ ĐỊNH tâm đường tròn ở (0, 0) để nằm giữa không gian - CHỈ 1 LẦN
        try:
            center = self.lookup_pt(objects[0])
            self.register_loss(
                f"center_at_origin_{center_name}",
                lambda c=center: c.x**2 + c.y**2,
                weight=50.0  # Moderate weight - không override constraints khác
            )
        except:
            pass

        if param_type_str == "incircle":
            # Incircle defined by triangle points
            self.circles.append((center_name, {'type': 'incircle', 'triangle': [p.val for p in args]}))
        elif param_type_str == "circumcircle":
            # Circumcircle defined by triangle points
            self.circles.append((center_name, {'type': 'circumcircle', 'triangle': [p.val for p in args]}))
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
            self.segments.append((p1_name, p2_name))

    def _process_line_parameter(self, param_type, objects, args):
        """Process line instructions - store for visualization"""
        # Line through 2 points: (line A B)
        if len(objects) >= 2:
            p1_name = objects[0].val if hasattr(objects[0], 'val') else str(objects[0])
            p2_name = objects[1].val if hasattr(objects[1], 'val') else str(objects[1])
            self.lines.append((p1_name, p2_name))

    def _make_distance_ndg_loss(self, point1, point2):
        """Create a loss function that encourages two points to be far apart.
        
        Returns a lambda that computes negative squared distance (for NDG minimization).
        Minimizing negative distance = maximizing distance (encourages separation).
        
        Args:
            point1: First Point object
            point2: Second Point object
            
        Returns:
            Callable that returns -distance² between the two points
        """
        return lambda: -((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def process_assertion(self, assertion):
        """Process assertion/constraint instructions"""
        # Assertions are handled separately - they add constraints to existing objects
        if self.verbosity:
            logger.info(f"Processing assertion: {assertion}")

        # Parse assertion type and apply constraints
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
            elif assertion.constraint_type == 'on_circle':
                self._add_on_circle_constraint(assertion.objects)
                # Sau khi thêm on-circle constraint, enforce góc tối thiểu
                self._enforce_minimum_angle_between_circle_points()
            
            
    def _add_angle_measure(self, points: list):
        """Store angle measure for display: angle ABC = degrees"""
        # DSL: (angle-measure A C B 110) 
        if len(points) < 4:
            logger.warning(f"Angle-measure needs 4 values (3 points + degrees), got {len(points)}")
            return
        
        p1_name = points[0].val  # A
        vertex_name = points[1].val  # C (đỉnh góc)
        p2_name = points[2].val  # B
        degrees = float(points[3].val) if hasattr(points[3], 'val') else float(points[3])  # 110
        
        # Store for later rendering
        self.angle_measures.append((vertex_name, p1_name, p2_name, degrees))
        if self.verbosity:
            logger.info(f"Added angle measure: angle {p1_name}{vertex_name}{p2_name} = {degrees}°")

    def _add_on_segment_constraint(self, points: list):
        """Add constraint: point lies on segment (between endpoints)"""
        if len(points) != 3:
            logger.warning(f"on_segment constraint needs 3 points (point, seg_p1, seg_p2), got {len(points)}")
            return
        
        point = self.lookup_pt(points[0])
        seg_p1 = self.lookup_pt(points[1])
        seg_p2 = self.lookup_pt(points[2])
        
        key = f"{points[0].val}_on_{points[1].val}{points[2].val}"
        self.register_loss(f"on_segment_{key}",
            lambda pt=point, p1=seg_p1, p2=seg_p2: self._point_on_segment_loss(pt, p1, p2),
            weight=50.0  # Strong enough to keep point on segment, balanced with other constraints
        )
        if self.verbosity:
            logger.info(f"Added on-segment constraint: {points[0].val} on {points[1].val}{points[2].val}")

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
        
        p1 = self.lookup_pt(points[0])
        p2 = self.lookup_pt(points[1])
        p3 = self.lookup_pt(points[2])
        p4 = self.lookup_pt(points[3])
        
        self.register_loss(
            f"equal_distance_{points[0].val}{points[1].val}_{points[2].val}{points[3].val}",
            lambda pt1=p1, pt2=p2, pt3=p3, pt4=p4: (self.dist(pt1, pt2) - self.dist(pt3, pt4))**2,
            weight=100.0
        )
        
        if self.verbosity:
            logger.info(f"Added equal-distance constraint: {points[0].val}{points[1].val} = {points[2].val}{points[3].val}")
        
    def _add_on_circle_constraint(self, points: list):
        if len(points) != 2:
            logger.warning(f"on-circle constraint needs 2 points (point, center), got {len(points)}")
            return
        
        point_name = points[0]
        center_name = points[1].val
        
        logger.info(f"Adding on-circle constraint: point={point_name.val}, center={center_name}")
        logger.info(f"Available circles: {self.circles}")
        
        # Find radius from circle metadata
        radius = None
        for circle_name, circle_info in self.circles:
            if circle_name == center_name:
                radius = circle_info.get('radius')
                break
        if radius is None:
            logger.warning(f"Circle {center_name} not found or has no radius")
            return
        
        center = self.lookup_pt(points[1])
        logger.info(f"Found radius={radius} for circle {center_name}")
        
        # Smart initialization: place point on circle far from existing points
        point = self.lookup_pt(point_name)
        logger.info(f"Smart init for {point_name.val}: point found={point is not None}")
        if point is not None:
            # Find existing points on same circle
            existing_angles = []
            logger.info(f"Current loss_fns keys: {list(self.loss_fns.keys())}")
            for loss_name in self.loss_fns:
                if loss_name.startswith(f"on_circle_") and loss_name.endswith(f"_{center_name}"):
                    other_point_name = loss_name.split("_")[2]
                    logger.info(f"Checking loss {loss_name}, extracted point name: {other_point_name}")
                    if other_point_name != point_name.val:
                        try:
                            other_pt = self.lookup_pt_by_name(other_point_name)
                            logger.info(f"  lookup_pt_by_name({other_point_name}) = {other_pt}")
                            if other_pt:
                                # Calculate angle from center
                                dx = other_pt.x - center.x
                                dy = other_pt.y - center.y
                                angle = torch.atan2(dy, dx)
                                existing_angles.append(angle.item())
                                logger.info(f"  Added angle {angle.item():.3f} rad")
                        except Exception as e:
                            logger.warning(f"  Failed to get angle for {other_point_name}: {e}")
                            pass
            
            # Choose angle in the largest gap between existing points
            import math
            logger.info(f"Found {len(existing_angles)} existing points on circle {center_name}: {existing_angles}")
            if existing_angles:
                # Normalize all angles to [0, 2π]
                existing_angles = [(a + 2*math.pi) % (2*math.pi) for a in existing_angles]
                existing_angles.sort()
                logger.info(f"Normalized and sorted angles: {existing_angles}")
                
                # Calculate gaps between consecutive angles (wrapping around)
                gaps = []
                for i in range(len(existing_angles)):
                    next_i = (i + 1) % len(existing_angles)
                    start = existing_angles[i]
                    end = existing_angles[next_i]
                    
                    if next_i == 0:  # Wrap around from last to first
                        gap_size = (2*math.pi - start) + end
                    else:
                        gap_size = end - start
                    
                    gaps.append((gap_size, start))
                
                # Place new point in middle of largest gap
                largest_gap = max(gaps, key=lambda x: x[0])
                start_angle = largest_gap[1]
                gap_size = largest_gap[0]
                # Add small offset (10°) from middle to avoid perfect symmetry
                offset = math.radians(10)  # 10 degrees offset
                new_angle = (start_angle + gap_size / 2 + offset) % (2*math.pi)
                logger.info(f"Largest gap: size={gap_size:.3f} rad, start={start_angle:.3f}, placing point at angle={new_angle:.3f} (with 10° offset)")
            else:
                # Random angle if no existing points
                new_angle = random.uniform(0, 2 * math.pi)
            
            # Initialize on circle at this angle
            init_x = center.x.item() + radius * math.cos(new_angle)
            init_y = center.y.item() + radius * math.sin(new_angle)
            point.x.data.fill_(init_x)
            point.y.data.fill_(init_y)
        
        # Use const() to avoid lambda closure issues
        radius_const = self.const(radius)
        self.register_loss(
            f"on_circle_{point_name.val}_{center_name}",
            lambda pt=point, c=center, r=radius_const: (self.dist(pt, c) - r)**2,
            weight=100000.0  
        )
    
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
        
        for loss_name in self.losses:
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
                        # 🔥 THÊM HARD CONSTRAINT (không phải NDG) để BUỘC không thẳng hàng
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
                                        logger.info(f"🔥 Added HARD chord constraint: {p1_name}-{center_name}-{p2_name} MUST NOT be collinear")
                        except Exception as e:
                            if self.verbosity:
                                logger.warning(f"Failed to add chord constraint: {e}")
    
    def _enforce_minimum_angle_between_circle_points(self):
        """
        Enforce minimum angle (45 degrees) between points on the same circle.
        This ensures chords are long enough and diagrams look good.
        """
        import math
        
        # Group points by circle
        circle_points_map = {}  # {center_name: [point_objs]}
        
        for circle_name, circle_info in self.circles:
            circle_points_map[circle_name] = []
        
        # Find all points with on_circle constraint
        for loss_name in self.loss_fns:
            if loss_name.startswith("on_circle_"):
                parts = loss_name.split("_")
                if len(parts) >= 4:  # on_circle_PointName_CenterName
                    point_name = parts[2]
                    center_name = "_".join(parts[3:])
                    
                    if center_name in circle_points_map:
                        try:
                            pt_obj = self.lookup_pt_by_name(point_name)
                            if pt_obj:
                                circle_points_map[center_name].append((point_name, pt_obj))
                        except Exception as ex:
                            logger.warning(f"Failed to lookup point {point_name}: {ex}")
        
        # For each circle with 2+ points, enforce minimum angle
        min_angle_rad = math.radians(60)  # 60 degrees minimum 
        min_cos = self.const(math.cos(min_angle_rad))
        
        for center_name, points_list in circle_points_map.items():
            if len(points_list) < 2:
                continue
            
            try:
                center = self.lookup_pt_by_name(center_name)
                if not center:
                    continue
                
                # Only create constraints for the newly added point (last in list)
                # to avoid re-creating constraints that already exist
                new_point_idx = len(points_list) - 1
                new_name, new_pt = points_list[new_point_idx]
                
                # Pair the new point with all existing points
                for i in range(new_point_idx):
                    name1, pt1 = points_list[i]
                    
                    constraint_name = f"min_angle_{center_name}_{name1}_{new_name}"
                    
                    # Register minimum angle constraint between existing point and new point
                    self.register_loss(
                        constraint_name,
                        self._make_circle_angle_constraint(pt1, new_pt, center, min_cos),
                        weight=10000.0  # Very high weight to prevent point overlap
                    )
            except Exception as e:
                import traceback
                logger.warning(f"Failed to enforce min angle for circle {center_name}: {e}")
    
    def _make_circle_angle_constraint(self, pt1: TorchPoint, pt2: TorchPoint, center: TorchPoint, min_cos_value):
        """
        Factory method to create a reusable constraint object for minimum angle between circle points.
        
        Args:
            pt1, pt2: The two points on the circle
            center: Center of the circle
            min_cos_value: Minimum allowed cosine value (higher cos = smaller angle)
            
        Returns:
            CircleAngleConstraint: Callable object that can be reused
        """
        return self.CircleAngleConstraint(self, pt1, pt2, center, min_cos_value)
    
    def lookup_pt_by_name(self, name: str):
        """Helper to lookup point by string name"""
        if name in self.name2pt:
            return self.name2pt[name]
        return None
        
    
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
                          lambda: self.parallel(p1, p2, p3, p4), weight=10.0)

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
                          lambda: self.perpendicular(p1, p2, p3, p4), weight=10.0)

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
            weight=10.0
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

    def regularize_points(self):
        """Add regularization to keep points near origin"""
        if len(self.name2pt) > 0:
            def compute_reg():
                norms = [self.norm(p) for p in self.name2pt.values()]
                return torch.stack(norms).mean()
            self.register_loss("regularization", compute_reg, weight=0.01)

    def make_points_distinct(self):
        pts = list(self.name2pt.values())
        if len(pts) < 2:
            return

        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                # Encourage d > 0.1
                self.register_ndg(f"distinct_{i}_{j}",
                                 lambda pi=pts[i], pj=pts[j]: self.dist(pi, pj), weight=0.1)

    def train(self, epochs: int = 1000, lr: float = 0.01):
        if not self.has_loss:
            return 0.0

        optimizer = optim.Adam(self.trainable_vars, lr=lr)
        if self.verbosity:
            logger.info(f"Optimization ({epochs}) Epochs")

        for i in range(epochs):
            optimizer.zero_grad()
            # Compute losses fresh at each iteration
            self.losses = {key: fn() for key, fn in self.loss_fns.items()}
            total_loss = sum(self.losses.values())
            total_loss.backward()

            optimizer.step()
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

    

    def solve_single(self, attempt_id=0):
        self.current_attempt = attempt_id
        self.preprocess()
        
        self._add_all_chord_ndgs()
        
        self.regularize_points()
        # self.make_points_distinct()

        loss = float('inf')
        if self.has_loss:
            loss = self.train(epochs=self.epochs, lr=self.learning_rate)

        return self.get_diagram(), loss
    
    def solve(self, n_tries=None):
        import random

        if n_tries is None:
            n_tries = self.n_tries

        best_loss = float('inf')
        best_diagram = None

        for attempt in range(n_tries):
            if attempt > 0:
                self._init_state()
                random.seed(self.seed + attempt)
                torch.manual_seed(self.seed + attempt)

            if self.verbosity and n_tries > 1:
                logger.info(f"\nAttempt {attempt + 1}/{n_tries}")
            try:
                diagram, loss = self.solve_single(attempt_id=attempt)

                if loss < self.eps:
                    if self.verbosity and n_tries > 1:
                        logger.success(f"Converged at attempt {attempt + 1} with loss {loss:.6f}")
                    return diagram

                if loss < best_loss:
                    best_loss = loss
                    best_diagram = diagram
                    if self.verbosity and n_tries > 1:
                        logger.info(f"New best loss: {loss:.6f}")

            except Exception as e:
                if self.verbosity:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                continue

        if self.verbosity and n_tries > 1:
            logger.info(f"\nBest loss after {n_tries} attempts: {best_loss:.6f}")

        return best_diagram if best_diagram is not None else self.get_diagram()

    def get_diagram(self):
        diagram = Diagram()
        # Convert points
        for name, pt in self.name2pt.items():
            x = pt.x.detach().cpu().item()
            y = pt.y.detach().cpu().item()
            geo_pt = GeometricPoint(x, y, name)
            diagram.add_point(name, geo_pt)

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
                    # Radius = distance from center to any triangle side
                    triangle_points = info['triangle']
                    p1 = self.name2pt[triangle_points[0]]
                    p2 = self.name2pt[triangle_points[1]]
                    radius = self.dist_to_line(center_pt, p1, p2).detach().cpu().item()
                    info = {**info, 'radius': radius}
                    
                elif info['type'] == 'circumcircle':
                    # Radius = distance from center to any triangle vertex
                    triangle_points = info['triangle']
                    p1 = self.name2pt[triangle_points[0]]
                    radius = self.dist(center_pt, p1).detach().cpu().item()
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
        
        return diagram