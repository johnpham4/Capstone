import torch
import torch.nn as nn
import torch.optim as optim
import random
from loguru import logger
import math

from src.services.diagram.models.instructions import Parameter, Assertion
from src.services.diagram.models.value_objects import Point
from src.services.diagram.models.types import QuadrilateralType, TriangleType, DiagramType
from src.services.diagram.initializer import Initializer
from src.services.diagram import geometry as geo
from src.services.diagram.geometry import TorchPoint
from src.services.diagram.converter import build_diagram

class Optimizer:
    def __init__(self, instructions, opts, verbosity=False):
        self.instructions = instructions
        self.opts = opts
        self.verbosity = verbosity
        self._init_state()  # Initialize all state variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _init_state(self):
        self.name2pt = {}
        self.name2line = {}
        self.all_points = []
        self.losses = {}
        self.loss_fns = {}
        self.ndgs = {}
        self.triangles_metadata = {}
        self.circles = []
        self.quadrilaterals_metadata = {}
        self.segments = []
        self.lines = []
        self.angle_equal_assertions = []
        self.angle_measures = []
        self.angle_bisectors_metadata = []
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

    def const(self, x):
        return torch.tensor(x, dtype=torch.float64, device=self.device)

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

    def _sample_quadrilateral_with_init(self, points: list, init_method, *args, noise: float = 0.05):
        """Generic quadrilateral sampler using Initializer methods"""
        assert len(points) == 4
        init_coords = init_method(*args)
        init_coords = Initializer.add_noise(init_coords, noise)

        pt_objs = [self.sample_uniform(p, init_coords=init_coords[i]) for i, p in enumerate(points)]
        names = [p.val for p in points]
        return pt_objs, names

    def sample_square(self, points: list):
        assert len(points) == 4
        pt_objs, names = self._sample_quadrilateral_with_init(points, Initializer.init_square, 1.0)
        p1, p2, p3, p4 = pt_objs

        # All sides equal + right angle
        self.register_loss(f"sq_eq_12_23_{names[0]}", lambda: geo.dist(p1, p2) - geo.dist(p2, p3), weight=10.0)
        self.register_loss(f"sq_eq_23_34_{names[0]}", lambda: geo.dist(p2, p3) - geo.dist(p3, p4), weight=10.0)
        self.register_loss(f"sq_eq_34_41_{names[0]}", lambda: geo.dist(p3, p4) - geo.dist(p4, p1), weight=10.0)
        self.register_loss(f"sq_right_B_{names[0]}", lambda: geo.dot_product(p1, p2, p3), weight=10.0)
        self.register_ndg(f"sq_area_{names[0]}", lambda: geo.cross_product_area(p1, p2, p3), weight=20.0)
        key = tuple(names)
        self.quadrilaterals_metadata[key] = {'type': QuadrilateralType.SQUARE, 'equal_sides': [(0,1),(1,2),(2,3),(3,0)]}
        return pt_objs

    def sample_rectangle(self, points: list):
        assert len(points) == 4
        pt_objs, names = self._sample_quadrilateral_with_init(points, Initializer.init_rectangle, 1.6, 1.0)
        p1, p2, p3, p4 = pt_objs

        # Three right angles
        self.register_loss(f"rect_right_B_{names[0]}", lambda: geo.dot_product(p1, p2, p3), weight=10.0)
        self.register_loss(f"rect_right_C_{names[0]}", lambda: geo.dot_product(p2, p3, p4), weight=10.0)
        self.register_loss(f"rect_right_D_{names[0]}", lambda: geo.dot_product(p3, p4, p1), weight=10.0)
        self.register_ndg(f"rect_area_{names[0]}", lambda: geo.cross_product_area(p1, p2, p3), weight=20.0)
        key = tuple(names)
        self.quadrilaterals_metadata[key] = {'type': QuadrilateralType.RECTANGLE, 'equal_sides': [(0,2),(1,3)]}
        return pt_objs

    def sample_parallelogram(self, points: list):
        assert len(points) == 4
        # Use scalene quad init for parallelogram (more general shape)
        pt_objs, names = self._sample_quadrilateral_with_init(points, Initializer.init_scalene_quadrilateral, 1.5)
        p1, p2, p3, p4 = pt_objs

        # Opposite sides parallel
        self.register_loss(f"para_vec_x_{names[0]}", lambda: (p2.x - p1.x) - (p3.x - p4.x), weight=10.0)
        self.register_loss(f"para_vec_y_{names[0]}", lambda: (p2.y - p1.y) - (p3.y - p4.y), weight=10.0)
        self.register_ndg(f"para_area_{names[0]}", lambda: geo.cross_product_area(p2, p1, p3), weight=20.0)
        key = tuple(names)
        self.quadrilaterals_metadata[key] = {'type': QuadrilateralType.PARALLELOGRAM}
        return pt_objs

    def sample_trapezoid(self, points: list):
        assert len(points) == 4
        pt_objs, names = self._sample_quadrilateral_with_init(points, Initializer.init_scalene_quadrilateral, 1.2)
        p1, p2, p3, p4 = pt_objs

        # One pair parallel
        self.register_loss(f"trap_para_{names[0]}",
                          lambda: (p2.x - p1.x) * (p3.y - p4.y) - (p2.y - p1.y) * (p3.x - p4.x), weight=10.0)
        self.register_ndg(f"trap_area_{names[0]}", lambda: geo.cross_product_area(p1, p2, p3), weight=20.0)
        self.register_ndg(f"trap_ndg_top_{names[0]}", lambda: geo.dist(p3, p4), weight=10.0)
        self.register_ndg(f"trap_ndg_bottom_{names[0]}", lambda: geo.dist(p1, p2), weight=10.0)
        key = tuple(names)
        self.quadrilaterals_metadata[key] = {'type': QuadrilateralType.TRAPEZOID}
        return pt_objs

    def sample_rhombus(self, points: list):
        assert len(points) == 4
        pt_objs, names = self._sample_quadrilateral_with_init(points, Initializer.init_rhombus, 1.0)
        p1, p2, p3, p4 = pt_objs

        # All sides equal + diagonals perpendicular
        self.register_loss(f"rhombus_eq_12_23_{names[0]}", lambda: geo.dist(p1, p2) - geo.dist(p2, p3), weight=10.0)
        self.register_loss(f"rhombus_eq_23_34_{names[0]}", lambda: geo.dist(p2, p3) - geo.dist(p3, p4), weight=10.0)
        self.register_loss(f"rhombus_eq_34_41_{names[0]}", lambda: geo.dist(p3, p4) - geo.dist(p4, p1), weight=10.0)
        self.register_loss(f"rhombus_diag_perp_{names[0]}",
                          lambda: (p3.x - p1.x) * (p4.x - p2.x) + (p3.y - p1.y) * (p4.y - p2.y), weight=10.0)
        self.register_ndg(f"rhombus_area_{names[0]}", lambda: geo.cross_product_area(p1, p2, p3), weight=20.0)
        self.register_ndg(f"rhombus_diag_ac_{names[0]}", lambda: geo.dist(p1, p3), weight=10.0)
        key = tuple(names)
        self.quadrilaterals_metadata[key] = {'type': QuadrilateralType.RHOMBUS, 'equal_sides': [(0,1),(1,2),(2,3),(3,0)]}
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
        else:
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
                              lambda ap=apex_pt, o0=other_pts[0], o1=other_pts[1]: geo.dist(ap, o0) - geo.dist(ap, o1),
                              weight=10.0)
            others = [i for i in range(3) if i != apex_idx]
            metadata['equal_sides'] = [(apex_idx, others[0]), (apex_idx, others[1])]

        if tri_type == 'right' or tri_type == 'right_isosceles':
            right_pt = pts[right_idx]
            other_pts = [pts[i] for i in range(3) if i != right_idx]
            self.register_loss(f"right_{points[0].val}_{points[1].val}_{points[2].val}",
                              lambda: geo.dot_product(other_pts[0], right_pt, other_pts[1]), weight=10.0)
            metadata['right_angle_at'] = right_idx

        if tri_type == 'equilateral':
            self.register_loss(f"equi_12_23_{points[0].val}",
                              lambda: geo.dist(p1, p2) - geo.dist(p2, p3), weight=10.0)
            self.register_loss(f"equi_23_31_{points[0].val}",
                              lambda: geo.dist(p2, p3) - geo.dist(p3, p1), weight=10.0)
            metadata['equal_sides'] = [(0, 1), (1, 2), (2, 0)]

        # Handle equal_angles constraint
        if equal_angles:
            metadata['equal_angles'] = equal_angles
            for idx1, idx2 in equal_angles:
                self.register_loss(
                    f"equal_angle_{idx1}_{idx2}_{points[0].val}",
                    lambda i1=idx1, i2=idx2, p=pts: geo.triangle_angle_difference_loss(p, i1, i2),
                    weight=10.0
                )

        self.register_ndg(f"tri_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda: geo.collinear(p1, p2, p3), weight=1.0)
        key = (points[0].val, points[1].val, points[2].val)
        self.triangles_metadata[key] = metadata
        return [p1, p2, p3]

    def _sample_general_quadrilateral(self, points: list):
        assert len(points) == 4
        pt_objs, names = self._sample_quadrilateral_with_init(
            points, Initializer.init_scalene_quadrilateral, 1.0
        )
        self.register_ndg(
            f"quad_area_{names[0]}",
            lambda: geo.cross_product_area(pt_objs[0], pt_objs[1], pt_objs[2]),
            weight=20.0,
        )
        key = tuple(names)
        self.quadrilaterals_metadata[key] = {'type': QuadrilateralType.GENERAL}
        return pt_objs

    def _define_projection(self, point_name, vertex_point, segment_points):
        assert len(segment_points) == 2

        foot = self.sample_uniform(point_name)
        vertex = self.lookup_pt(vertex_point)
        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])

        # Foot perpendicular to segment and lies on segment
        self.register_loss(f"perp_{point_name.val}",
                          lambda: geo.perpendicular(vertex, foot, p1, p2), weight=10.0)
        self.register_loss(f"on_seg_{point_name.val}",
                          lambda: geo.collinear(foot, p1, p2), weight=10.0)
        return foot

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
                          lambda: (geo.dist_to_line(incenter, p1, p2) - geo.dist_to_line(incenter, p2, p3))**2 +
                                  (geo.dist_to_line(incenter, p2, p3) - geo.dist_to_line(incenter, p3, p1))**2,
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
                          lambda: (geo.dist(circumcenter, p1) - geo.dist(circumcenter, p2))**2 +
                                  (geo.dist(circumcenter, p2) - geo.dist(circumcenter, p3))**2,
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
                          lambda: geo.perpendicular(p1, orthocenter, p2, p3)**2 +
                                  geo.perpendicular(p2, orthocenter, p1, p3)**2,
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
                          weight=100.0)

        # Constraint 2: Collinearity
        self.register_loss(f"midpoint_collinear_{point_name.val}",
                          lambda: geo.collinear(midpoint, p1, p2),
                          weight=100.0)

        return midpoint

    def parameter_on_seg(self, p, segment_points: list):
        assert len(segment_points) == 2
        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])
        P = self.sample_uniform(p)
        self.register_loss(f"on_seg_{p.val}",
                          lambda: geo.collinear(P, p1, p2), weight=50.0)
        return P

    def parameter_on_line(self, p, line_points):
        assert len(line_points) == 2
        p1 = self.lookup_pt(line_points[0])
        p2 = self.lookup_pt(line_points[1])

        # Simple initialization without analytical solution
        P = self.sample_uniform(p, save_name=False)

        def on_line_loss():
            line = geo.pp2lnf(p1, p2)
            return geo.on_line(P, line)**2
        self.register_loss(f"on_line_{p.val}", on_line_loss, weight=5.0)
        return self.register_pt(p, P)

    def _define_line_intersection(self, point_name, line1_points, line2_points):
        assert len(line1_points) == 2 and len(line2_points) == 2
        p1 = self.lookup_pt(line1_points[0])
        p2 = self.lookup_pt(line1_points[1])
        p3 = self.lookup_pt(line2_points[0])
        p4 = self.lookup_pt(line2_points[1])
        intersection = self.sample_uniform(point_name)

        self.register_loss(f"on_line1_{point_name.val}",
                          lambda: geo.collinear(intersection, p1, p2), weight=10.0)
        self.register_loss(f"on_line2_{point_name.val}",
                          lambda: geo.collinear(intersection, p3, p4), weight=10.0)
        return intersection

    def _define_angle_bisector(self, point_name, angle_points):
        assert len(angle_points) >= 3, "angle_bisector requires 3 points [B, A, C] where A is vertex"

        p1 = self.lookup_pt(angle_points[0])
        vertex = self.lookup_pt(angle_points[1])
        p2 = self.lookup_pt(angle_points[2])

        init_x = (vertex.x.item() + p1.x.item() + p2.x.item()) / 3
        init_y = (vertex.y.item() + p1.y.item() + p2.y.item()) / 3
        bisector_point = self.sample_uniform(point_name, init_coords=(init_x, init_y))

        self.angle_bisectors_metadata.append({
            'vertex': angle_points[1].val,
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
                          lambda: (geo.dist(point, p1) - geo.dist(point, p2))**2,
                          weight=10.0)
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
        p1 = self.name2pt[angle_points[0].val]
        p_vertex = self.name2pt[vertex.val]
        p2 = self.name2pt[angle_points[2].val]
        p_bisector = self.name2pt[bisector_point.val]

        key = f"{vertex.val}_{bisector_point.val}"
        # Bisector constraint: Equal angles
        self.register_loss(
            f"bisector_equal_angle_{key}",
            lambda pv=p_vertex, p_1=p1, p_2=p2, pb=p_bisector: geo.angle_bisector_equal_loss(pv, p_1, p_2, pb),
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
            param_type_str = "general"

        quad_samplers = {
            "square": self.sample_square,
            "rectangle": self.sample_rectangle,
            "parallelogram": self.sample_parallelogram,
            "trapezoid": self.sample_trapezoid,
            "rhombus": self.sample_rhombus,
        }

        sampler = quad_samplers.get(param_type_str)
        if sampler:
            sampler(objects)
        else:
            try:
                quadri_type = QuadrilateralType[param_type_str.upper()]
            except (KeyError, AttributeError):
                quadri_type = QuadrilateralType.GENERAL
            constraints = {'type': quadri_type}
            self._sample_general_quadrilateral(objects)

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
                self._enforce_minimum_angle_between_circle_points()


    def _add_angle_measure(self, points: list):
        """Store angle measure for display: angle ABC = degrees"""
        # DSL: (angle-measure A C B 110)
        if len(points) < 4:
            logger.warning(f"Angle-measure needs 4 values (3 points + degrees), got {len(points)}")
            return

        p1_name = points[0].val
        vertex_name = points[1].val
        p2_name = points[2].val
        degrees = float(points[3].val) if hasattr(points[3], 'val') else float(points[3])

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

        key = f"{points[0].val}_on_{points[1].val}{points[2].val}"
        self.register_loss(f"on_segment_{key}",
            lambda pt=point, p1=seg_p1, p2=seg_p2: geo.point_on_segment_loss(pt, p1, p2),
            weight=20.0
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
            lambda pt1=p1, pt2=p2, d=distance_value: (geo.dist(pt1, pt2) - d)**2,
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
            lambda pt1=p1, pt2=p2, pt3=p3, pt4=p4: (geo.dist(pt1, pt2) - geo.dist(pt3, pt4))**2,
            weight=100.0
        )

        if self.verbosity:
            logger.info(f"Added equal-distance constraint: {points[0].val}{points[1].val} = {points[2].val}{points[3].val}")

    def _add_on_circle_constraint(self, points: list):
        if len(points) != 2:
            logger.warning(f"on-circle constraint needs 2 points (point, center), got {len(points)}")
            return

        point = self.lookup_pt(points[0])
        center_name = points[1].val

        logger.info(f"Adding on-circle constraint: point={points[0].val}, center={center_name}")
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

        self.register_loss(
            f"center_at_origin_{center_name}",
            lambda c=center: c.x**2 + c.y**2,
            weight=100000.0,
        )

        radius_const = self.const(radius)
        self.register_loss(
            f"on_circle_{points[0].val}_{center_name}",
            lambda pt=point, c=center, r=radius_const: (geo.dist(pt, c) - r)**2,
            weight=100000.0,
        )

    def _add_all_chord_ndgs(self):
        """Add NDG constraints for all chords to prevent degenerate diameters."""
        circle_points: dict[str, list[str]] = {}

        for loss_name in self.loss_fns:
            if loss_name.startswith("on_circle_"):
                parts = loss_name.split("_")
                if len(parts) >= 4:
                    point_name = parts[2]
                    center_name = "_".join(parts[3:])
                    circle_points.setdefault(center_name, []).append(point_name)

        for center_name, point_names in circle_points.items():
            center = self.lookup_pt_by_name(center_name)
            if not center:
                continue

            for i in range(len(point_names)):
                for j in range(i + 1, len(point_names)):
                    p1_name = point_names[i]
                    p2_name = point_names[j]

                    has_segment = (
                        (p1_name, p2_name) in self.segments or
                        (p2_name, p1_name) in self.segments
                    )
                    if not has_segment:
                        continue

                    pt1 = self.lookup_pt_by_name(p1_name)
                    pt2 = self.lookup_pt_by_name(p2_name)
                    if not pt1 or not pt2:
                        continue

                    ndg_key = f"chord_ndg_{p1_name}_{center_name}_{p2_name}"
                    if ndg_key not in self.ndgs:
                        self.register_ndg(
                            ndg_key,
                            lambda pt1_=pt1, c=center, pt2_=pt2: geo.collinear(pt1_, c, pt2_),
                            weight=100.0,
                        )

    def _enforce_minimum_angle_between_circle_points(self):
        """Penalise pairs of on-circle points whose central angle < 45°."""
        circle_points_map: dict[str, list[tuple]] = {
            name: [] for name, _ in self.circles
        }

        for loss_name in self.loss_fns:
            if loss_name.startswith("on_circle_"):
                parts = loss_name.split("_")
                if len(parts) >= 4:
                    point_name = parts[2]
                    center_name = "_".join(parts[3:])
                    if center_name in circle_points_map:
                        pt_obj = self.lookup_pt_by_name(point_name)
                        if pt_obj:
                            circle_points_map[center_name].append((point_name, pt_obj))

        min_cos = math.cos(math.radians(45))

        for center_name, points_list in circle_points_map.items():
            if len(points_list) < 2:
                continue
            center = self.lookup_pt_by_name(center_name)
            if not center:
                continue

            for i in range(len(points_list)):
                for j in range(i + 1, len(points_list)):
                    name1, pt1 = points_list[i]
                    name2, pt2 = points_list[j]

                    def angle_constraint(p1=pt1, p2=pt2, c=center, mc=min_cos):
                        v1_x = p1.x - c.x
                        v1_y = p1.y - c.y
                        v2_x = p2.x - c.x
                        v2_y = p2.y - c.y
                        dot = v1_x * v2_x + v1_y * v2_y
                        mag1 = (v1_x**2 + v1_y**2)**0.5 + 1e-8
                        mag2 = (v2_x**2 + v2_y**2)**0.5 + 1e-8
                        cos_angle = dot / (mag1 * mag2)
                        penalty = cos_angle - mc
                        return (penalty if penalty > 0 else 0)**2

                    self.register_loss(
                        f"min_angle_{center_name}_{name1}_{name2}",
                        angle_constraint,
                        weight=50.0,
                    )

    def lookup_pt_by_name(self, name: str):
        """Helper to lookup point by string name"""
        return self.name2pt.get(name)


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
                          lambda: geo.parallel(p1, p2, p3, p4), weight=10.0)

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
                          lambda: geo.perpendicular(p1, p2, p3, p4), weight=10.0)

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
            lambda: geo.angle_diff_loss(p1, p2, p3, p4, p5, p6),
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
                norms = [geo.norm(p) for p in self.name2pt.values()]
                return torch.stack(norms).mean()
            self.register_loss("regularization", compute_reg, weight=0.01)

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
                if self.verbosity:
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
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                continue

        if self.verbosity and n_tries > 1:
            logger.info(f"\nBest loss after {n_tries} attempts: {best_loss:.6f}")

        return best_diagram if best_diagram is not None else self.get_diagram()

    def get_diagram(self):
        return build_diagram(self)