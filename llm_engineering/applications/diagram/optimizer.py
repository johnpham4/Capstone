import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple

from llm_engineering.domains.geometry.instructions import Parameter
from llm_engineering.domains.geometry.value_objects import Point, Line
from llm_engineering.domains.geometry.entities import GeometricPoint, Diagram
from llm_engineering.applications.diagram.initializer import Initializer

from loguru import logger

TorchPoint = namedtuple("TorchPoint", ["x", "y"])
LineSF = namedtuple("LineSF", ["a", "b", "c", "p1", "p2"])
LineNF = namedtuple("LineNF", ["n", "f"])

class Optimizer:
    def __init__(self, instructions, opts, verbosity=False):
        self.instructions = instructions
        self.opts = opts
        self.verbosity = verbosity

        self.name2pt = {}  # Point name -> TorchPoint (with tensors)
        self.name2line = {}  # Line name -> LineNF
        self.all_points = []  # All points for visualization

        self.losses = {}  # Loss values (for logging)
        self.loss_fns = {}  # Loss functions (for training)
        self.ndgs = {}  # Non-degeneracy conditions

        # Diagram metadata tracking
        self.triangles_metadata = {}  # (p1, p2, p3) -> {type, right_angle_at, equal_sides}
        self.circles = []  # [(center_name, radius_or_points)]
        self.segments = []  # [(p1_name, p2_name)]
        self.lines = []  # [(p1_name, p2_name)] for visualization
        self.line_objects = {}  # line_name -> LineNF

        # Unnamed point generation for auto-created intersections
        self.unnamed_point_counter = 0

        # Optimization parameters
        self.has_loss = False
        self.trainable_vars = []  # List of nn.Parameter objects

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def collinear(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint):
        # Use cross product: (p2-p1) × (p3-p1) = 0
        v1x = p2.x - p1.x
        v1y = p2.y - p1.y
        v2x = p3.x - p1.x
        v2y = p3.y - p1.y
        return v1x * v2y - v1y * v2x

    def dist_to_line(self, point: TorchPoint, p1: TorchPoint, p2: TorchPoint):
        """Distance from point to line defined by p1, p2"""
        line = self.pp2lnf(p1, p2)
        return torch.abs(self.on_line(point, line))

    def register_pt(self, p: TorchPoint, P, save_name=True):
        if save_name:
            assert p.val not in self.name2pt
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

    def sample_triangle(self, points: list, constraints: dict = None):
        assert len(points) == 3

        constraints = constraints or {}
        tri_type = constraints.get('type', 'scalene')
        apex_idx = constraints.get('apex_idx', 0)
        right_idx = constraints.get('right_idx', 0)

        # Smart initialization based on type
        if tri_type == 'isosceles':
            init_coords = Initializer.init_isoceles_triangle(apex_idx)
        elif tri_type == 'right':
            init_coords = Initializer.init_right_triangle(right_idx)
        elif tri_type == 'equilateral':
            init_coords = Initializer.init_equilateral_triangle()
        elif tri_type == 'right_isosceles':
            init_coords = Initializer.init_right_isoceles_triangle(right_idx)
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
            def right_loss():
                vec1_x = other_pts[0].x - right_pt.x
                vec1_y = other_pts[0].y - right_pt.y
                vec2_x = other_pts[1].x - right_pt.x
                vec2_y = other_pts[1].y - right_pt.y
                return vec1_x * vec2_x + vec1_y * vec2_y
            self.register_loss(f"right_{points[0].val}_{points[1].val}_{points[2].val}",
                              right_loss, weight=10.0)
            metadata['right_angle_at'] = right_idx

        if tri_type == 'equilateral':
            self.register_loss(f"equi_12_23_{points[0].val}",
                              lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=10.0)
            self.register_loss(f"equi_23_31_{points[0].val}",
                              lambda: self.dist(p2, p3) - self.dist(p3, p1), weight=10.0)
            metadata['equal_sides'] = [(0, 1), (1, 2), (2, 0)]

        # Non-degeneracy
        self.register_ndg(f"tri_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda a=p1, b=p2, c=p3: self.collinear(a, b, c), weight=1.0)

        # Track metadata
        key = (points[0].val, points[1].val, points[2].val)
        self.triangles_metadata[key] = metadata

        return [p1, p2, p3]

    def _define_projection(self, point_name, vertex_point, segment_points):

        assert len(segment_points) == 2

        foot = self.sample_uniform(point_name)
        vertex = self.lookup_pt(vertex_point)
        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])

        # Loss 1: perpendicular to segment
        def perpendicular_loss():
            vec_vf_x = foot.x - vertex.x
            vec_vf_y = foot.y - vertex.y
            vec_seg_x = p2.x - p1.x
            vec_seg_y = p2.y - p1.y
            dot = vec_vf_x * vec_seg_x + vec_vf_y * vec_seg_y
            return dot

        self.register_loss(f"perpendicular_{point_name.val}", perpendicular_loss, weight=10.0)
        self.register_loss(f"on_segment_{point_name.val}", lambda: self.collinear(foot, p1, p2), weight=10.0)

        return foot

    def _define_centroid(self, point_name, triangle_points):
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        # Create learnable point
        centroid = self.sample_uniform(point_name)

        # Constraint: must be at centroid position
        def centroid_loss():
            expected_x = (p1.x + p2.x + p3.x) / 3
            expected_y = (p1.y + p2.y + p3.y) / 3
            return (centroid.x - expected_x)**2 + (centroid.y - expected_y)**2

        self.register_loss(f"centroid_{point_name.val}", centroid_loss, weight=10.0)
        return centroid

    def _define_incenter(self, point_name, triangle_points):
        """Define incenter - equal distance to all sides"""
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        # Create named point with smart init
        init_coords = Initializer.init_triangle_incircle()
        init_coords = Initializer.add_noise(init_coords)
        incenter = self.sample_uniform(point_name, init_coords=init_coords[3])

        # Constraint: equal distance to all sides
        def incircle_loss():
            d1 = self.dist_to_line(incenter, p1, p2)
            d2 = self.dist_to_line(incenter, p2, p3)
            d3 = self.dist_to_line(incenter, p3, p1)
            return (d1 - d2)**2 + (d2 - d3)**2

        self.register_loss(f"incenter_{point_name.val}", incircle_loss, weight=10.0)
        return incenter

    def _define_circumcenter(self, point_name, triangle_points):
        """Define circumcenter - equal distance to all vertices"""
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        # Create named point with smart init
        init_coords = Initializer.init_triangle_circumcircle(radius=1.0)
        init_coords = Initializer.add_noise(init_coords, noise_scale=0.02)
        circumcenter = self.sample_uniform(point_name, init_coords=init_coords[3])

        # Constraint: equal distance to all vertices
        def circumcircle_loss():
            d1 = self.dist(circumcenter, p1)
            d2 = self.dist(circumcenter, p2)
            d3 = self.dist(circumcenter, p3)
            return (d1 - d2)**2 + (d2 - d3)**2

        self.register_loss(f"circumcenter_{point_name.val}", circumcircle_loss, weight=10.0)
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

        # Constraint: altitudes intersect at orthocenter
        def orthocenter_loss():
            # Altitude from p1 perpendicular to p2-p3
            vec_h1_x = p1.x - orthocenter.x
            vec_h1_y = p1.y - orthocenter.y
            vec_23_x = p3.x - p2.x
            vec_23_y = p3.y - p2.y
            perp1 = vec_h1_x * vec_23_x + vec_h1_y * vec_23_y

            # Altitude from p2 perpendicular to p1-p3
            vec_h2_x = p2.x - orthocenter.x
            vec_h2_y = p2.y - orthocenter.y
            vec_13_x = p3.x - p1.x
            vec_13_y = p3.y - p1.y
            perp2 = vec_h2_x * vec_13_x + vec_h2_y * vec_13_y

            return perp1**2 + perp2**2

        self.register_loss(f"orthocenter_{point_name.val}", orthocenter_loss, weight=10.0)
        return orthocenter

    def _define_midpoint(self, point_name, segment_points):
        assert len(segment_points) == 2

        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])

        midpoint = self.sample_uniform(point_name)

        def midpoint_loss():
            expected_x = (p1.x + p2.x) / 2
            expected_y = (p1.y + p2.y) / 2
            return (midpoint.x - expected_x)**2 + (midpoint.y - expected_y)**2

        self.register_loss(f"midpoint_{point_name.val}", midpoint_loss, weight=5.0)
        self.register_loss(f"on_segment_mid_{point_name.val}", lambda: self.collinear(midpoint, p1, p2), weight=10.0)
        return midpoint

    def parameter_on_seg(self, p, segment_points: list):
        assert len(segment_points) == 2

        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])

        P = self.sample_uniform(p)

        self.register_loss(f"on_seg_{p.val}", lambda: self.collinear(P, p1, p2), weight=10.0)

        return P

    def parameter_on_line(self, p, line_points):
        assert len(line_points) == 2

        p1 = self.lookup_pt(line_points[0])
        p2 = self.lookup_pt(line_points[1])

        # Create a free point
        P = self.sample_uniform(p, save_name=False)

        # Constrain it to be on the line - recompute line each iteration
        def on_line_loss():
            line = self.pp2lnf(p1, p2)
            return self.on_line(P, line)**2

        self.register_loss(f"on_line_{p.val}", on_line_loss, weight=10.0)

        return self.register_pt(p, P)

    def _define_line_intersection(self, point_name, line1_points, line2_points):
        """Define intersection point of two lines"""
        assert len(line1_points) == 2 and len(line2_points) == 2

        p1 = self.lookup_pt(line1_points[0])
        p2 = self.lookup_pt(line1_points[1])
        p3 = self.lookup_pt(line2_points[0])
        p4 = self.lookup_pt(line2_points[1])

        # Create learnable intersection point
        intersection = self.sample_uniform(point_name)

        # Constraint: point must be on both lines - recompute lines each iteration
        def intersection_loss():
            line1 = self.pp2lnf(p1, p2)
            line2 = self.pp2lnf(p3, p4)
            dist1 = self.on_line(intersection, line1)
            dist2 = self.on_line(intersection, line2)
            return dist1**2 + dist2**2

        self.register_loss(f"intersection_{point_name.val}", intersection_loss, weight=10.0)
        return intersection

    def _define_perpendicular_bisector_point(self, point_name, segment_points):
        """Define a point that lies on the perpendicular bisector of a segment"""
        assert len(segment_points) == 2

        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])

        # Create learnable point
        point = self.sample_uniform(point_name)

        # Constraint: equidistant from both endpoints
        def perp_bisector_loss():
            d1 = self.dist(point, p1)
            d2 = self.dist(point, p2)
            return (d1 - d2)**2

        self.register_loss(f"perp_bisector_{point_name.val}", perp_bisector_loss, weight=10.0)
        return point

    def process_instruction(self, instr):
        from llm_engineering.domains.geometry.instructions import Assertion

        if isinstance(instr, Parameter):
            self.process_parameter(instr)
        elif isinstance(instr, Assertion):
            self.process_assertion(instr)


    def process_parameter(self, instr):
        from llm_engineering.domains.geometry.types import TriangleType, DiagramType

        diagram_type = instr.diagram_type
        param_type = instr.param_type
        objects = instr.objects
        args = instr.args

        # Dispatch based on diagram type
        if diagram_type == DiagramType.TRIANGLE:
            self._process_triangle_parameter(param_type, objects, args)
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

    def _process_triangle_parameter(self, param_type, objects, args):

        from llm_engineering.domains.geometry.types import TriangleType

        # Handle TriangleType enum
        if isinstance(param_type, TriangleType):
            param_type_str = str(param_type).split('.')[-1].lower()
        else:
            param_type_str = str(param_type).lower() if param_type else ""

        # Build constraints dict
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

        # Single unified call
        self.sample_triangle(objects, constraints)

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
        elif param_type_str == "coords" or param_type_str == "":
            self.sample_uniform(objects[0])
        else:
            if self.verbosity:
                logger.warning(f"Unsupported point construction: {param_type_str}")

    def _process_circle_parameter(self, param_type, objects, args):
        """Process circle instructions and track them"""
        param_type_str = str(param_type).lower() if param_type else ""
        center_name = objects[0].val if hasattr(objects[0], 'val') else str(objects[0])

        if param_type_str == "incircle":
            # Incircle defined by triangle points
            self.circles.append((center_name, {'type': 'incircle', 'triangle': [p.val for p in args]}))
        elif param_type_str == "circumcircle":
            # Circumcircle defined by triangle points
            self.circles.append((center_name, {'type': 'circumcircle', 'triangle': [p.val for p in args]}))
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

    def _add_parallel_constraint(self, segments):
        """Add parallel constraint between two segments"""
        if len(segments) != 4:  # Need 4 points for 2 segments
            logger.warning(f"Parallel constraint needs 4 points (2 segments), got {len(segments)}")
            return

        p1 = self.lookup_pt(segments[0])
        p2 = self.lookup_pt(segments[1])
        p3 = self.lookup_pt(segments[2])
        p4 = self.lookup_pt(segments[3])

        # Parallel: direction vectors proportional
        # (p2-p1) × (p4-p3) = 0 (cross product = 0)
        def parallel_loss():
            dx1 = p2.x - p1.x
            dy1 = p2.y - p1.y
            dx2 = p4.x - p3.x
            dy2 = p4.y - p3.y
            # Cross product should be zero
            cross = dx1 * dy2 - dy1 * dx2
            return cross

        seg1_name = f"{segments[0].val}_{segments[1].val}"
        seg2_name = f"{segments[2].val}_{segments[3].val}"
        self.register_loss(f"parallel_{seg1_name}_{seg2_name}", parallel_loss, weight=10.0)

    def _add_perpendicular_constraint(self, segments):
        """Add perpendicular constraint between two segments"""
        if len(segments) != 4:
            logger.warning(f"Perpendicular constraint needs 4 points (2 segments), got {len(segments)}")
            return

        p1 = self.lookup_pt(segments[0])
        p2 = self.lookup_pt(segments[1])
        p3 = self.lookup_pt(segments[2])
        p4 = self.lookup_pt(segments[3])

        # Perpendicular: dot product = 0
        def perpendicular_loss():
            dx1 = p2.x - p1.x
            dy1 = p2.y - p1.y
            dx2 = p4.x - p3.x
            dy2 = p4.y - p3.y
            # Dot product should be zero
            dot = dx1 * dx2 + dy1 * dy2
            return dot

        seg1_name = f"{segments[0].val}_{segments[1].val}"
        seg2_name = f"{segments[2].val}_{segments[3].val}"
        self.register_loss(f"perpendicular_{seg1_name}_{seg2_name}", perpendicular_loss, weight=10.0)


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

        # Add small penalty for points being too close
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

    def solve(self):
        # Preprocess instructions
        self.preprocess()

        # Add regularization
        self.regularize_points()
        # self.make_points_distinct()

        # Optimize
        if self.has_loss:
            self.train(epochs=self.opts.get('epochs', 1000),
                      lr=self.opts.get('learning_rate', 0.01))

        return self.get_diagram()

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

                diagram.add_triangle(p1, p2, p3, equal_sides, right_angle_at)

        # Add circles
        for center_name, info in self.circles:
            if center_name in diagram.points:
                center = diagram.points[center_name]
                diagram.add_circle(center, info)

        # Add segments
        for p1_name, p2_name in self.segments:
            if p1_name in diagram.points and p2_name in diagram.points:
                p1 = diagram.points[p1_name]
                p2 = diagram.points[p2_name]
                diagram.add_segment(p1, p2)

        # Add lines
        for p1_name, p2_name in self.lines:
            if p1_name in diagram.points and p2_name in diagram.points:
                p1 = diagram.points[p1_name]
                p2 = diagram.points[p2_name]
                # Store line as tuple for rendering
                line_name = f"line_{p1_name}_{p2_name}"
                diagram.add_line(line_name, (p1, p2))

        return diagram