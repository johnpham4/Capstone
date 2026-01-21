import torch
import torch.nn as nn
import torch.optim as optim
import random

from collections import namedtuple
from llm_engineering.domains.geometry.instructions import Parameter, Assertion
from llm_engineering.domains.geometry.value_objects import Point, Line
from llm_engineering.domains.geometry.entities import GeometricPoint, Diagram
from llm_engineering.domains.geometry.types import QuadrilateralType, TriangleType, DiagramType
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize state
        self._init_state()

    def _init_state(self):
        """Initialize/reset optimizer state for new attempt"""
        self.name2pt = {}  # Point name -> TorchPoint (with tensors)
        self.name2line = {}  # Line name -> LineNF
        self.all_points = []  # All points for visualization

        self.losses = {}  # Loss values (for logging)
        self.loss_fns = {}  # Loss functions (for training)
        self.ndgs = {}  # Non-degeneracy conditions

        # Diagram metadata tracking
        self.triangles_metadata = {}  # (p1, p2, p3) -> {type, right_angle_at, equal_sides}
        self.circles = []  # [(center_name, radius_or_points)]
        self.quadrilaterals_metadata = {}
        self.segments = []  # [(p1_name, p2_name)]
        self.lines = []  # [(p1_name, p2_name)] for visualization
        self.line_objects = {}  # line_name -> LineNF
        self.angle_equal_assertions = []  # [(p1, p2, p3, p4, p5, p6)] for angle ABC = angle DEF

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
        # Add attempt_id to variable name to avoid conflicts across attempts
        if hasattr(self, 'current_attempt'):
            name = f"{name}_a{self.current_attempt}"

        if init_value is not None:
            val = torch.tensor([init_value], dtype=torch.float64, device=self.device)
        else:
            val = torch.empty(1, dtype=torch.float64, device=self.device).uniform_(lo, hi)
        param = nn.Parameter(val)
        self.trainable_vars.append(param)
        return param.squeeze()

    def generate_unnamed_point_name(self):
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
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        n_x = -dy
        n_y = dx
        n_norm = torch.sqrt(n_x**2 + n_y**2)
        if n_norm < 1e-6: n_norm = 1e-6
        n_x = n_x / n_norm
        n_y = n_y / n_norm
        if n_y < 0:
            n_x = -n_x
            n_y = -n_y
        r = n_x * p1.x + n_y * p1.y
        n = self.get_point(n_x, n_y)
        return LineNF(n, r)

    def on_line(self, p: TorchPoint, line: LineNF):
        # line in normal form: n · p - f = 0
        return line.n.x * p.x + line.n.y * p.y - line.f

    def collinear(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint):
        v1x = p2.x - p1.x
        v1y = p2.y - p1.y
        v2x = p3.x - p1.x
        v2y = p3.y - p1.y
        return v1x * v2y - v1y * v2x

    def dist_to_line(self, point: TorchPoint, p1: TorchPoint, p2: TorchPoint):
        line = self.pp2lnf(p1, p2)
        return torch.abs(self.on_line(point, line))

    def register_pt(self, p: TorchPoint, P, save_name=True):
        if save_name:
            assert p.val not in self.name2pt
            self.name2pt[p.val] = P

            if p.val not in self.name2pt:
                self.name2pt[p.val] = P
            else:
                self.name2pt[p.val] = P
        self.all_points.append(P)
        return P

    def register_loss(self, key, val_fn, weight: float = 1.0):
        if key in self.loss_fns: key = f"{key}_{len(self.loss_fns)}"
        self.loss_fns[key] = lambda w=weight, fn=val_fn: w * (fn() ** 2).mean()
        self.has_loss = True

    def register_ndg(self, key, val_fn, weight=1.0):
        if key in self.ndgs: key = f"{key}_{len(self.ndgs)}"
        loss_fn = lambda w=weight, fn=val_fn: w * torch.exp(-(fn() ** 2) * 50).mean()
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
        if save_name and p.val in self.name2pt:
            return self.name2pt[p.val]
        if init_coords is not None:
            x = self.mkvar(f"{p.val}_x", lo, hi, init_value=init_coords[0])
            y = self.mkvar(f"{p.val}_y", lo, hi, init_value=init_coords[1])
        else:
            x = self.mkvar(f"{p.val}_x", lo, hi)
            y = self.mkvar(f"{p.val}_y", lo, hi)
        P = self.get_point(x, y)
        return self.register_pt(p, P, save_name)


    def sample_square(self, points: list):
        assert len(points) == 4
        noise = 0.05
        init_positions = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        pt_objs = []
        for i, p_def in enumerate(points):
            ix, iy = init_positions[i]
            nx = ix + random.uniform(-noise, noise)
            ny = iy + random.uniform(-noise, noise)
            p_obj = self.sample_uniform(p_def, init_coords=(nx, ny))
            pt_objs.append(p_obj)
        p1, p2, p3, p4 = pt_objs[0], pt_objs[1], pt_objs[2], pt_objs[3]
        names = [p.val for p in points]
        self.quadrilaterals.append(tuple(names))

        self.register_loss(f"sq_eq_12_23_{names[0]}", lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=10.0)
        self.register_loss(f"sq_eq_23_34_{names[0]}", lambda: self.dist(p2, p3) - self.dist(p3, p4), weight=10.0)
        self.register_loss(f"sq_eq_34_41_{names[0]}", lambda: self.dist(p3, p4) - self.dist(p4, p1), weight=10.0)
        
        def dot_product(pa, pb, pc):
            v1x, v1y = pa.x - pb.x, pa.y - pb.y
            v2x, v2y = pc.x - pb.x, pc.y - pb.y
            return v1x * v2x + v1y * v2y
        self.register_loss(f"sq_right_B_{names[0]}", lambda: dot_product(p1, p2, p3), weight=10.0)
        
        def area_check():
             v1x, v1y = p2.x - p1.x, p2.y - p1.y
             v2x, v2y = p3.x - p1.x, p3.y - p1.y
             return v1x * v2y - v1y * v2x
        self.register_ndg(f"sq_area_{names[0]}", area_check, weight=20.0)
        return pt_objs

    def sample_rectangle(self, points: list):
        assert len(points) == 4
        noise = 0.05
        init_positions = [(-0.8, -0.5), (0.8, -0.5), (0.8, 0.5), (-0.8, 0.5)]
        pt_objs = []
        for i, p_def in enumerate(points):
            ix, iy = init_positions[i]
            nx = ix + random.uniform(-noise, noise)
            ny = iy + random.uniform(-noise, noise)
            p_obj = self.sample_uniform(p_def, init_coords=(nx, ny))
            pt_objs.append(p_obj)
        p1, p2, p3, p4 = pt_objs[0], pt_objs[1], pt_objs[2], pt_objs[3]
        names = [p.val for p in points]
        self.quadrilaterals.append(tuple(names))

        def dot_product(pa, pb, pc):
            v1x, v1y = pa.x - pb.x, pa.y - pb.y
            v2x, v2y = pc.x - pb.x, pc.y - pb.y
            return v1x * v2x + v1y * v2y
        self.register_loss(f"rect_right_B_{names[0]}", lambda: dot_product(p1, p2, p3), weight=10.0)
        self.register_loss(f"rect_right_C_{names[0]}", lambda: dot_product(p2, p3, p4), weight=10.0)
        self.register_loss(f"rect_right_D_{names[0]}", lambda: dot_product(p3, p4, p1), weight=10.0)
        
        def area_check():
             v1x, v1y = p2.x - p1.x, p2.y - p1.y
             v2x, v2y = p3.x - p1.x, p3.y - p1.y
             return v1x * v2y - v1y * v2x
        self.register_ndg(f"rect_area_{names[0]}", area_check, weight=20.0)
        return pt_objs

    def sample_parallelogram(self, points: list):
        assert len(points) == 4
        noise = 0.05
        init_positions = [(-1.5, -0.5), (0.5, -0.5), (1.5, 0.5), (-0.5, 0.5)]
        pt_objs = []
        for i, p_def in enumerate(points):
            ix, iy = init_positions[i]
            nx = ix + random.uniform(-noise, noise)
            ny = iy + random.uniform(-noise, noise)
            p_obj = self.sample_uniform(p_def, init_coords=(nx, ny))
            pt_objs.append(p_obj)
        p1, p2, p3, p4 = pt_objs[0], pt_objs[1], pt_objs[2], pt_objs[3]
        names = [p.val for p in points]
        self.quadrilaterals.append(tuple(names))

        def vec_diff_x():
            v_ab_x = p2.x - p1.x
            v_dc_x = p3.x - p4.x
            return v_ab_x - v_dc_x
        def vec_diff_y():
            v_ab_y = p2.y - p1.y
            v_dc_y = p3.y - p4.y
            return v_ab_y - v_dc_y
        self.register_loss(f"para_vec_x_{names[0]}", vec_diff_x, weight=10.0)
        self.register_loss(f"para_vec_y_{names[0]}", vec_diff_y, weight=10.0)

        def area_check():
             v1x, v1y = p1.x - p2.x, p1.y - p2.y
             v2x, v2y = p3.x - p2.x, p3.y - p2.y
             return v1x * v2y - v1y * v2x
        self.register_ndg(f"para_area_{names[0]}", area_check, weight=20.0)
        return pt_objs

    def sample_trapezoid(self, points: list):
        assert len(points) == 4
        noise = 0.05
        init_positions = [(-1.0, -0.5), (1.0, -0.5), (0.5, 0.5), (-0.5, 0.5)]
        pt_objs = []
        for i, p_def in enumerate(points):
            ix, iy = init_positions[i]
            nx = ix + random.uniform(-noise, noise)
            ny = iy + random.uniform(-noise, noise)
            p_obj = self.sample_uniform(p_def, init_coords=(nx, ny))
            pt_objs.append(p_obj)
        p1, p2, p3, p4 = pt_objs[0], pt_objs[1], pt_objs[2], pt_objs[3]
        names = [p.val for p in points]
        self.quadrilaterals.append(tuple(names))

        def parallel_loss():
            v_ab_x = p2.x - p1.x
            v_ab_y = p2.y - p1.y
            v_dc_x = p3.x - p4.x
            v_dc_y = p3.y - p4.y
            return v_ab_x * v_dc_y - v_ab_y * v_dc_x
        self.register_loss(f"trap_para_{names[0]}", parallel_loss, weight=10.0)

        def area_check():
             v1x, v1y = p2.x - p1.x, p2.y - p1.y
             v2x, v2y = p3.x - p2.x, p3.y - p2.y
             return v1x * v2y - v1y * v2x
        self.register_ndg(f"trap_area_{names[0]}", area_check, weight=20.0)
        self.register_ndg(f"trap_ndg_top_{names[0]}", lambda: self.dist(p3, p4), weight=10.0)
        self.register_ndg(f"trap_ndg_bottom_{names[0]}", lambda: self.dist(p1, p2), weight=10.0)
        return pt_objs

    def sample_rhombus(self, points: list):
        assert len(points) == 4, "Rhombus must have 4 points"
        noise = 0.05
        # Khởi tạo hình thoi (dạng Diamond)
        init_positions = [(0.0, -0.8), (0.5, 0.0), (0.0, 0.8), (-0.5, 0.0)]

        pt_objs = []
        for i, p_def in enumerate(points):
            ix, iy = init_positions[i]
            nx = ix + random.uniform(-noise, noise)
            ny = iy + random.uniform(-noise, noise)
            p_obj = self.sample_uniform(p_def, init_coords=(nx, ny))
            pt_objs.append(p_obj)

        p1, p2, p3, p4 = pt_objs[0], pt_objs[1], pt_objs[2], pt_objs[3]
        names = [p.val for p in points]
        self.quadrilaterals.append(tuple(names))

        # 4 cạnh bằng nhau
        self.register_loss(f"rhombus_eq_12_23_{names[0]}", lambda: self.dist(p1, p2) - self.dist(p2, p3), weight=10.0)
        self.register_loss(f"rhombus_eq_23_34_{names[0]}", lambda: self.dist(p2, p3) - self.dist(p3, p4), weight=10.0)
        self.register_loss(f"rhombus_eq_34_41_{names[0]}", lambda: self.dist(p3, p4) - self.dist(p4, p1), weight=10.0)

        # Hai đường chéo vuông góc: AC vuông góc BD
        def diagonals_perp():
            v_ac_x = p3.x - p1.x
            v_ac_y = p3.y - p1.y
            v_bd_x = p4.x - p2.x
            v_bd_y = p4.y - p2.y
            return v_ac_x * v_bd_x + v_ac_y * v_bd_y
        self.register_loss(f"rhombus_diag_perp_{names[0]}", diagonals_perp, weight=10.0)

        # Chống bẹt
        def area_check():
             v1x, v1y = p2.x - p1.x, p2.y - p1.y
             v2x, v2y = p3.x - p2.x, p3.y - p2.y
             return v1x * v2y - v1y * v2x
        self.register_ndg(f"rhombus_area_{names[0]}", area_check, weight=20.0)
        self.register_ndg(f"rhombus_diag_ac_{names[0]}", lambda: self.dist(p1, p3), weight=10.0)

        return pt_objs


    def sample_triangle(self, points: list, constraints: dict = None):
        assert len(points) == 3
        constraints = constraints or {}
        tri_type = constraints.get('type', 'scalene')
        apex_idx = constraints.get('apex_idx', 0)
        right_idx = constraints.get('right_idx', 0)
        equal_angles = constraints.get('equal_angles')
        

        if tri_type == 'isosceles':
            init_coords = Initializer.init_isoceles_triangle(apex_idx)
        elif tri_type == 'right':
            init_coords = Initializer.init_right_triangle(right_idx)
        elif tri_type == 'equilateral':
            init_coords = Initializer.init_equilateral_triangle()
        elif tri_type == 'right_isosceles':
            init_coords = Initializer.init_right_isoceles_triangle(right_idx)
        elif equal_angles:
            idx1, idx2 = equal_angles[0]
            apex_idx = 3 - idx1 - idx2  # Đỉnh thứ 3 (0+1+2=3)
            init_coords = Initializer.init_isoceles_triangle(apex_idx)
        else:
            # Scalene triangle init
            init_coords = Initializer.init_scalene_triangle()

        init_coords = Initializer.add_noise(init_coords)

        # Create points

        p1 = self.sample_uniform(points[0], init_coords=init_coords[0])
        p2 = self.sample_uniform(points[1], init_coords=init_coords[1])
        p3 = self.sample_uniform(points[2], init_coords=init_coords[2])
        pts = [p1, p2, p3]

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
        
        # Handle equal_angles constraint
        if equal_angles:
            metadata['equal_angles'] = equal_angles
            for idx1, idx2 in equal_angles:
                # Enforce angle equality using cosine similarity
                def angle_loss(i1=idx1, i2=idx2):
                    # Angle at vertex i1
                    v1_prev = pts[(i1-1)%3]
                    v1_curr = pts[i1]
                    v1_next = pts[(i1+1)%3]
                    
                    # Angle at vertex i2
                    v2_prev = pts[(i2-1)%3]
                    v2_curr = pts[i2]
                    v2_next = pts[(i2+1)%3]
                    
                    # Cosine of angle at i1
                    vec1a_x = v1_prev.x - v1_curr.x
                    vec1a_y = v1_prev.y - v1_curr.y
                    vec1b_x = v1_next.x - v1_curr.x
                    vec1b_y = v1_next.y - v1_curr.y
                    
                    dot1 = vec1a_x * vec1b_x + vec1a_y * vec1b_y
                    norm1a = torch.sqrt(vec1a_x**2 + vec1a_y**2 + 1e-8)
                    norm1b = torch.sqrt(vec1b_x**2 + vec1b_y**2 + 1e-8)
                    cos1 = dot1 / (norm1a * norm1b + 1e-8)
                    
                    # Cosine of angle at i2
                    vec2a_x = v2_prev.x - v2_curr.x
                    vec2a_y = v2_prev.y - v2_curr.y
                    vec2b_x = v2_next.x - v2_curr.x
                    vec2b_y = v2_next.y - v2_curr.y
                    
                    dot2 = vec2a_x * vec2b_x + vec2a_y * vec2b_y
                    norm2a = torch.sqrt(vec2a_x**2 + vec2a_y**2 + 1e-8)
                    norm2b = torch.sqrt(vec2b_x**2 + vec2b_y**2 + 1e-8)
                    cos2 = dot2 / (norm2a * norm2b + 1e-8)
                    
                    return cos1 - cos2
                
                self.register_loss(f"equal_angle_{idx1}_{idx2}_{points[0].val}", 
                                 angle_loss, weight=10.0)

        # Non-degeneracy
        self.register_ndg(f"tri_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda a=p1, b=p2, c=p3: self.collinear(a, b, c), weight=1.0)

        # Track metadata
        key = (points[0].val, points[1].val, points[2].val)
        self.triangles_metadata[key] = metadata
        
        logger.info(f"Triangle {key} metadata: {metadata}")

        return [p1, p2, p3]

    #tu giac
    def sample_quadrilateral(self, points, constraints=None, init_coords=None):
        assert len(points) == 4

        constraints = constraints or {}
        quadri_type = constraints.get('type', QuadrilateralType.GENERAL)

        if quadri_type == QuadrilateralType.SQUARE:
            init_coords = Initializer.init_square(side=1.0)
        elif quadri_type == QuadrilateralType.RECTANGLE:
            init_coords = Initializer.init_rectangle(width=1.0, height=0.7)
        elif quadri_type == QuadrilateralType.RHOMBUS:
            init_coords = Initializer.init_rhombus(scale=1.0)
        else:
            init_coords = Initializer.init_scalene_quadrilateral(scale=1.0)
        init_coords = Initializer.add_noise(init_coords)

        #create points
        p1 = self.sample_uniform(points[0], init_coords=init_coords[0])
        p2 = self.sample_uniform(points[1], init_coords=init_coords[1])
        p3 = self.sample_uniform(points[2], init_coords=init_coords[2])
        p4 = self.sample_uniform(points[3], init_coords=init_coords[3])
        pts = [p1, p2, p3, p4]

        metadata = {'type': quadri_type}
        if quadri_type in [QuadrilateralType.SQUARE, QuadrilateralType.RECTANGLE]:

        # Constraint: 4 goc vuong
            def angle_at_A():
                v1x = pts[3].x - pts[0].x # DA
                v1y = pts[3].y - pts[0].y
                v2x = pts[1].x - pts[0].x # AB
                v2y = pts[1].y - pts[0].y
                return v1x * v2x + v1y * v2y # tich vo huong = 0

            def angle_at_B():
                v1x = pts[0].x - pts[1].x # BA
                v1y = pts[0].y - pts[1].y
                v2x = pts[2].x - pts[1].x # BC
                v2y = pts[2].y - pts[1].y
                return v1x * v2x + v1y * v2y

            def angle_at_C():
                v1x = pts[1].x - pts[2].x # CB
                v1y = pts[1].y - pts[2].y
                v2x = pts[3].x - pts[2].x # CD
                v2y = pts[3].y - pts[2].y
                return v1x * v2x + v1y * v2y

            def angle_at_D():
                v1x = pts[2].x - pts[3].x # DC
                v1y = pts[2].y - pts[3].y
                v2x = pts[0].x - pts[3].x # DA
                v2y = pts[0].y - pts[3].y
                return v1x * v2x + v1y * v2y

            self.register_loss(f"angle_A", angle_at_A, weight=100.0)
            self.register_loss(f"angle_B", angle_at_B, weight=100.0)
            self.register_loss(f"angle_C", angle_at_C, weight=100.0)
            self.register_loss(f"angle_D", angle_at_D, weight=100.0)

        if quadri_type == QuadrilateralType.SQUARE:
            # constraint: 4 canh bang nhau
            def equal_sides_constraint():
                d01 = self.dist(pts[0], pts[1]) # AB
                d12 = self.dist(pts[1], pts[2]) # BC
                d23 = self.dist(pts[2], pts[3]) # CD
                d30 = self.dist(pts[3], pts[0]) # DA
                avg=(d01 + d12 + d23 + d30) / 4.0
                return (d01 - avg)**2 + (d12 - avg)**2 + (d23 - avg)**2 + (d30 - avg)**2

            self.register_loss(f"square_equal_sides", equal_sides_constraint, weight=100.0)
            self.register_ndg(f"square_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda a=pts[0], b=pts[1], c=pts[2]: self.collinear(a, b, c),
                         weight=1.0)
            metadata['equal_sides'] = [(0, 1), (1, 2), (2, 3), (3, 0)]

        elif quadri_type == QuadrilateralType.RECTANGLE:
            # constraint: canh doi bang nhau
            def equal_opposite_sides_constraint():
                d01 = self.dist(pts[0], pts[1]) # AB
                d12 = self.dist(pts[1], pts[2]) # BC
                d23 = self.dist(pts[2], pts[3]) # CD
                d30 = self.dist(pts[3], pts[0]) # DA
                return (d01 - d23)**2 + (d12 - d30)**2

            self.register_loss(f"rect_opposite_sides_{points[0].val}_{points[1].val}_{points[2].val}_{points[3].val}",
            equal_opposite_sides_constraint,weight=10.0)
            metadata['equal_sides'] = [(0, 2), (1, 3)]

            self.register_ndg(f"rect_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                     lambda a=pts[0], b=pts[1], c=pts[2]: self.collinear(a, b, c),
                     weight=1.0)

        elif quadri_type == QuadrilateralType.RHOMBUS:
            def equal_all_sides_constraint():
                d01 = self.dist(pts[0], pts[1])
                d12 = self.dist(pts[1], pts[2])
                d23 = self.dist(pts[2], pts[3])
                d30 = self.dist(pts[3], pts[0])
                avg = (d01 + d12 + d23 + d30) / 4.0
                return (d01 - avg)**2 + (d12 - avg)**2 + (d23 - avg)**2 + (d30 - avg)**2

            self.register_loss(f"rhombus_equal_sides_{points[0].val}", equal_all_sides_constraint, weight=10.0)
            metadata['equal_sides'] = [(0, 1), (1, 2), (2, 3), (3, 0)]

            self.register_ndg(f"rhombus_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                     lambda a=pts[0], b=pts[1], c=pts[2]: self.collinear(a, b, c),
                     weight=1.0)

        # Track metadata
        key = tuple(p.val for p in points)
        self.quadrilaterals_metadata[key] = metadata
        return [p1, p2, p3, p4]


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

    def _define_intersection(self, point_name, segment1_points, segment2_points):
        assert len(segment1_points) == 2 and len(segment2_points) == 2

        p1 = self.lookup_pt(segment1_points[0])
        p2 = self.lookup_pt(segment1_points[1])
        p3 = self.lookup_pt(segment2_points[0])
        p4 = self.lookup_pt(segment2_points[1])

        # Initialize near the center of the bounding box for better convergence
        init_x = (p1.x.item() + p2.x.item() + p3.x.item() + p4.x.item()) / 4
        init_y = (p1.y.item() + p2.y.item() + p3.y.item() + p4.y.item()) / 4
        intersection = self.sample_uniform(point_name, init_coords=(init_x, init_y))

        def intersection_loss():
            line1 = self.pp2lnf(p1, p2)
            line2 = self.pp2lnf(p3, p4)
            dist1 = self.on_line(intersection, line1)
            dist2 = self.on_line(intersection, line2)
            return dist1**2 + dist2**2

        # Use very high weight to ensure precise intersection (same as midpoint)
        self.register_loss(f"intersection_{point_name.val}", intersection_loss, weight=50.0)
        # Add collinear constraints to ensure point lies on both lines
        self.register_loss(f"collinear_line1_{point_name.val}", lambda: self.collinear(intersection, p1, p2), weight=50.0)
        self.register_loss(f"collinear_line2_{point_name.val}", lambda: self.collinear(intersection, p3, p4), weight=50.0)
        return intersection
        

    def _define_centroid(self, point_name, triangle_points):
        assert len(triangle_points) == 3
        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])
        centroid = self.sample_uniform(point_name)
        def centroid_loss():
            expected_x = (p1.x + p2.x + p3.x) / 3
            expected_y = (p1.y + p2.y + p3.y) / 3
            return (centroid.x - expected_x)**2 + (centroid.y - expected_y)**2
        self.register_loss(f"centroid_{point_name.val}", centroid_loss, weight=10.0)
        return centroid

    def _define_incenter(self, point_name, triangle_points):
        assert len(triangle_points) == 3
        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])
        init_coords = Initializer.init_triangle_incircle()
        init_coords = Initializer.add_noise(init_coords)
        incenter = self.sample_uniform(point_name, init_coords=init_coords[3])
        def incircle_loss():
            d1 = self.dist_to_line(incenter, p1, p2)
            d2 = self.dist_to_line(incenter, p2, p3)
            d3 = self.dist_to_line(incenter, p3, p1)
            return (d1 - d2)**2 + (d2 - d3)**2
        self.register_loss(f"incenter_{point_name.val}", incircle_loss, weight=10.0)
        return incenter

    def _define_circumcenter(self, point_name, triangle_points):
        assert len(triangle_points) == 3
        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])
        init_coords = Initializer.init_triangle_circumcircle(radius=1.0)
        init_coords = Initializer.add_noise(init_coords, noise_scale=0.02)
        circumcenter = self.sample_uniform(point_name, init_coords=init_coords[3])
        def circumcircle_loss():
            d1 = self.dist(circumcenter, p1)
            d2 = self.dist(circumcenter, p2)
            d3 = self.dist(circumcenter, p3)
            return (d1 - d2)**2 + (d2 - d3)**2
        self.register_loss(f"circumcenter_{point_name.val}", circumcircle_loss, weight=10.0)
        return circumcenter

    def _define_orthocenter(self, point_name, triangle_points):
        assert len(triangle_points) == 3
        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])
        init_coords = Initializer.init_right_triangle_with_orthocenter()
        init_coords = Initializer.add_noise(init_coords)
        orthocenter = self.sample_uniform(point_name, init_coords=init_coords[3])
        def orthocenter_loss():
            vec_h1_x = p1.x - orthocenter.x
            vec_h1_y = p1.y - orthocenter.y
            vec_23_x = p3.x - p2.x
            vec_23_y = p3.y - p2.y
            perp1 = vec_h1_x * vec_23_x + vec_h1_y * vec_23_y
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
        P = self.sample_uniform(p, save_name=False)
        def on_line_loss():
            line = self.pp2lnf(p1, p2)
            return self.on_line(P, line)**2
        self.register_loss(f"on_line_{p.val}", on_line_loss, weight=10.0)
        return self.register_pt(p, P)

    def _define_line_intersection(self, point_name, line1_points, line2_points):
        assert len(line1_points) == 2 and len(line2_points) == 2
        p1 = self.lookup_pt(line1_points[0])
        p2 = self.lookup_pt(line1_points[1])
        p3 = self.lookup_pt(line2_points[0])
        p4 = self.lookup_pt(line2_points[1])
        intersection = self.sample_uniform(point_name)
        def intersection_loss():
            line1 = self.pp2lnf(p1, p2)
            line2 = self.pp2lnf(p3, p4)
            dist1 = self.on_line(intersection, line1)
            dist2 = self.on_line(intersection, line2)
            return dist1**2 + dist2**2

        self.register_loss(f"intersection_{point_name.val}", intersection_loss, weight=10.0)
        return intersection

    def _define_centroid(self, point_name, triangle_points):
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        # Centroid is deterministic: (A+B+C)/3 - compute directly with small learnable offset
        init_x = (p1.x.item() + p2.x.item() + p3.x.item()) / 3
        init_y = (p1.y.item() + p2.y.item() + p3.y.item()) / 3
        centroid = self.sample_uniform(point_name, init_coords=(init_x, init_y))

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

        # Init near centroid (incenter is inside triangle, close to centroid)
        init_x = (p1.x.item() + p2.x.item() + p3.x.item()) / 3
        init_y = (p1.y.item() + p2.y.item() + p3.y.item()) / 3
        incenter = self.sample_uniform(point_name, init_coords=(init_x, init_y))

        # Constraint: equal distance to all sides
        def incircle_loss():
            d1 = self.dist_to_line(incenter, p1, p2)
            d2 = self.dist_to_line(incenter, p2, p3)
            d3 = self.dist_to_line(incenter, p3, p1)
            return (d1 - d2)**2 + (d2 - d3)**2 + (d3 - d1)**2  # Add third comparison

        self.register_loss(f"incenter_{point_name.val}", incircle_loss, weight=50.0)  # Increased from 10.0
        return incenter

    def _define_circumcenter(self, point_name, triangle_points):
        """Define circumcenter - equal distance to all vertices"""
        assert len(triangle_points) == 3

        p1 = self.lookup_pt(triangle_points[0])
        p2 = self.lookup_pt(triangle_points[1])
        p3 = self.lookup_pt(triangle_points[2])

        # Init near centroid (circumcenter can be inside or outside triangle)
        init_x = (p1.x.item() + p2.x.item() + p3.x.item()) / 3
        init_y = (p1.y.item() + p2.y.item() + p3.y.item()) / 3
        circumcenter = self.sample_uniform(point_name, init_coords=(init_x, init_y))

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

        # Init near centroid (orthocenter can be inside or outside triangle)
        init_x = (p1.x.item() + p2.x.item() + p3.x.item()) / 3
        init_y = (p1.y.item() + p2.y.item() + p3.y.item()) / 3
        orthocenter = self.sample_uniform(point_name, init_coords=(init_x, init_y))

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

        self.register_loss(f"midpoint_{point_name.val}", midpoint_loss, weight=20.0)  # Increased from 5.0
        self.register_loss(f"on_segment_mid_{point_name.val}", lambda: self.collinear(midpoint, p1, p2), weight=20.0)  # Increased from 10.0
        return midpoint

    def _define_angle_bisector(self, point_name, angle_points):
        """
        Define angle bisector point
        DSL format: (angle_bisector A B C) where A is vertex, angle BAC is bisected
        Returns D on BC such that angle BAD = angle CAD
        """
        assert len(angle_points) >= 3, "angle_bisector requires at least 3 points [A, B, C]"
        
        vertex = self.lookup_pt(angle_points[0])  # A (đỉnh góc)
        p1 = self.lookup_pt(angle_points[1])  # B
        p2 = self.lookup_pt(angle_points[2])  # C
        
        # Smart initialization: midpoint of BC (works well for isosceles)
        init_x = (p1.x.item() + p2.x.item()) / 2
        init_y = (p1.y.item() + p2.y.item()) / 2
        bisector_point = self.sample_uniform(point_name, init_coords=(init_x, init_y))
        
        # Save metadata for rendering
        if not hasattr(self, 'angle_bisectors_metadata'):
            self.angle_bisectors_metadata = []
        
        self.angle_bisectors_metadata.append({
            'vertex': vertex if isinstance(vertex, str) else angle_points[0].val,
            'bisector_point': point_name.val,
            'angle_points': [p.val for p in angle_points]
        })
        
        # Apply constraints using existing _process_angle_bisector
        # Pass Point objects for compatibility
        self._process_angle_bisector(angle_points[0], point_name, angle_points)
        
        return bisector_point

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


    def _define_perpendicular_bisector_point(self, point_name, segment_points):
        assert len(segment_points) == 2
        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])
        point = self.sample_uniform(point_name)
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
        diagram_type = instr.diagram_type
        param_type = instr.param_type
        objects = instr.objects
        args = instr.args

        if diagram_type == DiagramType.TRIANGLE:
            self._process_triangle_parameter(param_type, objects, args)
        elif diagram_type == DiagramType.QUADRILATERAL:
             p_type_str = str(param_type).split('.')[-1].lower()
             if p_type_str == "square":
                 self.sample_square(objects)
             elif p_type_str == "rectangle":
                 self.sample_rectangle(objects)
             elif p_type_str == "parallelogram":
                 self.sample_parallelogram(objects)
             elif p_type_str == "trapezoid":
                 self.sample_trapezoid(objects)
             elif p_type_str == "rhombus": # [NEW]
                 self.sample_rhombus(objects)
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
        # Trong tam giác ABC, AD là phân giác góc A
        # vertex = A, bisector_point = D, angle_points = [A, B, C]
        
        p_vertex = self.name2pt[vertex.val]  # A
        p_bisector = self.name2pt[bisector_point.val]  # D
        
        p1 = self.name2pt[angle_points[1].val]  # B 
        p2 = self.name2pt[angle_points[2].val]  # C
        
        # constraint 1: D nam tren canh BC
        def on_segment_loss():
            # Vector BD và BC cùng phương (D nằm trên đường thẳng BC)
            bd_x = p_bisector.x - p1.x  
            bd_y = p_bisector.y - p1.y
            bc_x = p2.x - p1.x  
            bc_y = p2.y - p1.y
            cross = bd_x * bc_y - bd_y * bc_x  # Cross product = 0 → cùng phương
            
            # D nằm giữa B và C: BD = t * BC với 0 <= t <= 1
            bc_len_sq = bc_x**2 + bc_y**2
            t = (bd_x * bc_x + bd_y * bc_y) / (bc_len_sq + 1e-8)
            
            # Phạt nếu t < 0 hoặc t > 1
            between_penalty = torch.relu(-t) + torch.relu(t - 1)
        
            return cross**2 + 10.0 * between_penalty
        
        # constrain 2: goc BAD = goc CAD
        def equal_angle_loss():
            # Cosine of angle BAD
            ab_x = p1.x - p_vertex.x # vector AB
            ab_y = p1.y - p_vertex.y
            ad_x = p_bisector.x - p_vertex.x # vector AD
            ad_y = p_bisector.y - p_vertex.y
            dot1 = ab_x * ad_x + ab_y * ad_y # tich vo huong
            
            ab_norm = torch.sqrt(ab_x**2 + ab_y**2 + 1e-8)
            ad_norm = torch.sqrt(ad_x**2 + ad_y**2 + 1e-8)
            cos_bad = dot1 / (ab_norm * ad_norm + 1e-8)
            
            # Cosine of angle CAD
            ac_x = p2.x - p_vertex.x # vector AC
            ac_y = p2.y - p_vertex.y
            dot2 = ac_x * ad_x + ac_y * ad_y
            
            ac_norm = torch.sqrt(ac_x**2 + ac_y**2 + 1e-8)
            cos_cad = dot2 / (ac_norm * ad_norm + 1e-8)
            return (cos_bad - cos_cad)**2

        #constraint 3: BD/DC = AB/AC (dinh ly phan giac)
        def ratio_loss():
            # bd length
            bd_x = p_bisector.x - p1.x
            bd_y = p_bisector.y - p1.y
            bd_len = torch.sqrt(bd_x**2 + bd_y**2 + 1e-8)
            
            # dc length
            dc_x = p2.x - p_bisector.x
            dc_y = p2.y - p_bisector.y
            dc_len = torch.sqrt(dc_x**2 + dc_y**2 + 1e-8)
            
            # ab length
            ab_x = p1.x - p_vertex.x
            ab_y = p1.y - p_vertex.y
            ab_len = torch.sqrt(ab_x**2 + ab_y**2 + 1e-8)
            
            # ac length
            ac_x = p2.x - p_vertex.x
            ac_y = p2.y - p_vertex.y
            ac_len = torch.sqrt(ac_x**2 + ac_y**2 + 1e-8)
            
            # bd/dc - ab/ac = 0
            ratio_bd_dc = bd_len / (dc_len + 1e-8)
            ratio_ab_ac = ab_len / (ac_len + 1e-8)
            return (ratio_bd_dc - ratio_ab_ac)**2
        
        key = f"{vertex.val}_{bisector_point.val}"
        self.register_loss(f"bisector_on_segment_{key}", on_segment_loss, weight=10.0)
        self.register_loss(f"bisector_equal_angle_{key}", equal_angle_loss, weight=10.0)
        self.register_loss(f"bisector_ratio_{key}", ratio_loss, weight=5.0)    
        

    def _process_triangle_parameter(self, param_type, objects, args):
        if isinstance(param_type, TriangleType):
            param_type_str = str(param_type).split('.')[-1].lower()
        else:
            param_type_str = str(param_type).lower() if param_type else ""

        constraints = {}
        if param_type_str == "isosceles":
            constraints['type'] = 'isosceles'
            if args:
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
        elif param_type_str in ["equal_angles", "equal-angles"]:
            # DSL: (triangle (A B C) (equal_angles 0 1))
            constraints['type'] = 'scalene'
            if args and len(args) >= 2:
                idx1 = int(str(args[0]))
                idx2 = int(str(args[1]))
                constraints['equal_angles'] = [(idx1, idx2)]
                logger.info(f"Setting equal_angles constraint in process_triangle_parameter: {constraints['equal_angles']}")
        else:
            constraints['type'] = 'scalene'

        # Single unified call
        self.sample_triangle(objects, constraints)

    def _process_quadrilateral_parameter(self, param_type, objects, args):
        """Process quadrilateral parameters"""
        # Convert param_type to QuadrilateralType enum
        if param_type:
            param_type_str = str(param_type).upper()
            try:
                quadri_type = QuadrilateralType[param_type_str]
            except KeyError:
                quadri_type = QuadrilateralType.GENERAL
        else:
            quadri_type = QuadrilateralType.GENERAL

        logger.info(f"Quadrilateral type: {quadri_type}")

        constraints = {'type': quadri_type}

        self.sample_quadrilateral(objects, constraints)

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
        elif param_type_str == "bisector":
            # DSL: (define D point (bisector A B C))
            # args = [A, B, C] where A is vertex
            self._define_angle_bisector(objects[0], args)
        elif param_type_str == "projection":
            self._define_projection(objects[0], args[0], args[1:])
        elif param_type_str == "intersection":
            if len(args) >= 4:
                self._define_intersection(objects[0], args[0:2], args[2:4])
            else:
                logger.warning(f"intersection requires 4 points, got {len(args)}")

        elif param_type_str == "segment":
            self.parameter_on_seg(objects[0], args)
        elif param_type_str == "line":
            self.parameter_on_line(objects[0], args)
        elif param_type_str in ["inter-ll", "inter_ll"]:
            # args should be 4 points: line1_p1, line1_p2, line2_p1, line2_p2
            if len(args) >= 4:
                self._define_intersection(objects[0], args[0:2], args[2:4])
            else:
                if self.verbosity:
                    logger.warning(f"inter-ll requires 4 points, got {len(args)}")
            if len(args) >= 4:
                self._define_line_intersection(objects[0], args[0:2], args[2:4])
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

    def _add_parallel_constraint(self, segments):
        """Add parallel constraint between two segments"""
        if len(segments) != 4:  # Need 4 points for 2 segments
            logger.warning(f"Parallel constraint needs 4 points (2 segments), got {len(segments)}")
            return

        p1 = self.lookup_pt(segments[0])
        p2 = self.lookup_pt(segments[1])
        p3 = self.lookup_pt(segments[2])
        p4 = self.lookup_pt(segments[3])

        def parallel_loss():
            # Direction vector 1
            dx1 = p2.x - p1.x
            dy1 = p2.y - p1.y
            len1 = torch.sqrt(dx1**2 + dy1**2) + 1e-8

            # Direction vector 2
            dx2 = p4.x - p3.x
            dy2 = p4.y - p3.y
            len2 = torch.sqrt(dx2**2 + dy2**2) + 1e-8

            # Normalize
            dx1_norm = dx1 / len1
            dy1_norm = dy1 / len1
            dx2_norm = dx2 / len2
            dy2_norm = dy2 / len2

            # Cross product of normalized vectors should be zero
            cross = dx1_norm * dy2_norm - dy1_norm * dx2_norm

            # Also check dot product is close to ±1 (same or opposite direction)
            dot = dx1_norm * dx2_norm + dy1_norm * dy2_norm
            angle_error = 1 - torch.abs(dot)  # Should be 0 if parallel

            return cross + angle_error

        seg1_name = f"{segments[0].val}_{segments[1].val}"
        seg2_name = f"{segments[2].val}_{segments[3].val}"
        self.register_loss(f"parallel_{seg1_name}_{seg2_name}", parallel_loss, weight=50.0)  # Increased from 10.0

    def _add_perpendicular_constraint(self, segments):
        """Add perpendicular constraint between two segments"""
        if len(segments) != 4:
            logger.warning(f"Perpendicular constraint needs 4 points (2 segments), got {len(segments)}")
            return

        p1 = self.lookup_pt(segments[0])
        p2 = self.lookup_pt(segments[1])
        p3 = self.lookup_pt(segments[2])
        p4 = self.lookup_pt(segments[3])

        def perpendicular_loss():
            dx1 = p2.x - p1.x
            dy1 = p2.y - p1.y
            dx2 = p4.x - p3.x
            dy2 = p4.y - p3.y
            dot = dx1 * dx2 + dy1 * dy2
            return dot

        seg1_name = f"{segments[0].val}_{segments[1].val}"
        seg2_name = f"{segments[2].val}_{segments[3].val}"
        self.register_loss(f"perpendicular_{seg1_name}_{seg2_name}", perpendicular_loss, weight=10.0)

    def _add_angle_equal_constraint(self, points):
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

        def angle_equal_loss():
            v1_1x = p1.x - p2.x
            v1_1y = p1.y - p2.y
            v1_2x = p3.x - p2.x
            v1_2y = p3.y - p2.y
            len1_1 = torch.sqrt(v1_1x**2 + v1_1y**2 + 1e-8)
            len1_2 = torch.sqrt(v1_2x**2 + v1_2y**2 + 1e-8)
            cos1 = (v1_1x * v1_2x + v1_1y * v1_2y) / (len1_1 * len1_2)

            v2_1x = p4.x - p5.x
            v2_1y = p4.y - p5.y
            v2_2x = p6.x - p5.x
            v2_2y = p6.y - p5.y
            len2_1 = torch.sqrt(v2_1x**2 + v2_1y**2 + 1e-8)
            len2_2 = torch.sqrt(v2_2x**2 + v2_2y**2 + 1e-8)
            cos2 = (v2_1x * v2_2x + v2_1y * v2_2y) / (len2_1 * len2_2)

            return (cos1 - cos2)**2

        angle1_name = f"{points[0].val}_{points[1].val}_{points[2].val}"
        angle2_name = f"{points[3].val}_{points[4].val}_{points[5].val}"
        self.register_loss(f"angle_equal_{angle1_name}_{angle2_name}", angle_equal_loss, weight=10.0)
        
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
        if len(self.name2pt) > 0:
            def compute_reg():
                norms = [self.norm(p) for p in self.name2pt.values()]
                return torch.stack(norms).mean()
            self.register_loss("regularization", compute_reg, weight=0.001)  

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
                # Log loss breakdown
                breakdown = ", ".join([f"{k}: {v.item():.6f}" for k, v in self.losses.items()])
                logger.info(f"Iteration {i:4d}: Total = {total_loss.item():.6f} | {breakdown}")

            # Early stopping - stricter threshold
            if total_loss.item() < 1e-8:  # Reduced from 1e-6
                if self.verbosity >= 0:
                    logger.info(f"Converged at iteration {i} with loss {total_loss.item():.6f}")
                break

        final_loss = total_loss.item()

        if self.verbosity:
            logger.info(f"Final loss {final_loss:.6f}")
            self.log_losses()

        return final_loss


    def log_losses(self):
        if len(self.loss_fns) == 0: return
        if not self.losses: self.losses = {key: fn() for key, fn in self.loss_fns.items()}
        logger.info("\n Loss breakdown")
        for key, loss in self.losses.items():
            logger.info(f"{key:30s}: {loss.item():.6f}")

    def solve_single(self, attempt_id=0):
        """Solve with single initialization attempt"""
        self.current_attempt = attempt_id

        # Preprocess instructions
        self.preprocess()

        # Add regularization
        self.regularize_points()

        # Optimize
        loss = float('inf')
        if self.has_loss:
            loss = self.train(epochs=self.opts.get('epochs', 1000),
                            lr=self.opts.get('learning_rate', 0.01))

        return self.get_diagram(), loss

    def solve(self, n_tries=None):
        """Solve with multiple initialization attempts"""
        import random

        # Default to 1 try for backward compatibility
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
            
            # Check all points exist
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
        return diagram