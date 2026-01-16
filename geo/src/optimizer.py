from primitives import *
from instructions import *
import torch
from collections import namedtuple
from loguru import logger
import torch.optim as optim
from diagram import Diagram


TorchPoint = namedtuple("TorchPoint", ["x", "y"])
LineSF = namedtuple("LineSF", ["a", "b", "c", "p1", "p2"])
LineNF = namedtuple("LineNF", ["n", "f"])

class Optimizer:
    def __init__(self, instructions, opts, verbose=False):
        self.instructions = instructions
        self.opts = opts
        self.verbose = verbose
        
        self.name2pt = {}
        self.name2line = {} # Line name -> LineNF
        self.all_points = []  # list of TorchPoint      
        
        self.losses = {}  # Loss values (for logging)
        self.loss_fns = {}  # Loss functions (for training)
        self.ndgs = {}  # Non-degeneracy conditions
        self.goals = {}  # Goal constraints to achieve
        self.iso_triangles = {} # Lưu thông tin tam giác cân: key -> apex_idx
        self.quadrilateral_type = None
        
        #optimize 
        self.has_loss = False
        self.trainable_val = [] # list of nn.Parameter
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def get_point(self, x, y):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float64, device=self.device)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float64, device=self.device)
        return TorchPoint(x, y)
    
    def lookup_pt(self, p):
        if isinstance(p, Point):
            if p.val in self.name2pt:
                return self.name2pt[p.val]
            else:
                raise RuntimeError(f"Point {p.val} not found")
        else:
            raise RuntimeError(f"Invalid point type: {type(p)}")
        
    def paramerter_on_seg(self, p, segment_points: list):
        assert len(segment_points) == 2

        p1 = self.lookup_pt(segment_points[0])
        p2 = self.lookup_pt(segment_points[1])

        # Create parameter t in [0, 1]
        t = self.mkvar(f"{p.val}_t", 0.0, 1.0)

        # Interpolate: p = p1 + t * (p2 - p1)
        x = p1.x + t * (p2.x - p1.x)
        y = p1.y + t * (p2.y - p1.y)

        P = self.get_point(x, y)
        return self.register_pt(p, P)
    

    def register_pt(self, p: TorchPoint, P, save_name=True):
        if save_name:
            assert p.val not in self.name2pt
            self.name2pt[p.val] = P

        self.all_points.append(P)
        return P
    
    def register_line(self, line_name, line_nf): # line_nf: LineNF -> Line normalized form
        assert line_name not in self.name2line
        self.name2line[line_name] = line_nf
        return line_nf
        
        
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
        
            
    def mkvar(self, name=None, lo=-1.0, hi=1.0):
        val = torch.empty(1, dtype = torch.float64, device = self.device)
        val.uniform_(lo, hi)
        param = torch.nn.Parameter(val)
        self.trainable_val.append(param)  
        return param.squeeze()
    
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
    
    def sample_uniform(self, p, lo=-1.0, hi=1.0, save_name=True):
        """Sample a point uniformly in a box"""
        x = self.mkvar(f"{p.val}_x", lo, hi)
        y = self.mkvar(f"{p.val}_y", lo, hi)
        P = self.get_point(x, y)
        return self.register_pt(p, P, save_name)
            
            
    def sample_triangle(self, points): # tam giác thường
        # Implement optimization logic for isosceles triangle here
        assert len(points) == 3  # triangle defined by 3 points example (param (A B C) triangle (iso-tri A))
        
        # create points for the triangle vertices
        point_1 = self.sample_uniform(points[0])
        point_2 = self.sample_uniform(points[1])
        point_3 = self.sample_uniform(points[2])
        
        self.register_ndg(f"tri_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda a=point_1, b=point_2, c=point_3: self.collinear(a, b, c), weight=1.0)
        return [point_1, point_2, point_3]
    
    def sample_right_triangle(self, points, right_angle_point): # tam giác vuông
        # ví dụ: (param (A B C) (right-tri C)) --> vuông tại C
        assert len(points) == 3
        right_idx = None
        for i, p in enumerate(points):
            if p.val == right_angle_point.val:
                right_idx = i
                break
        if right_idx is None:
            right_idx = 0  # default to first point if right angle point not found
        
        point_1 = self.sample_uniform(points[0])
        point_2 = self.sample_uniform(points[1])
        point_3 = self.sample_uniform(points[2])
        
        pts = [point_1, point_2, point_3]
        right_point = pts[right_idx]
        other_pts = [pts[i] for i in range(3) if i != right_idx]
        
        def perpendicular_constraint():
            v1x = other_pts[0].x - right_point.x
            v1y = other_pts[0].y - right_point.y
            v2x = other_pts[1].x - right_point.x
            v2y = other_pts[1].y - right_point.y
            dot_product = v1x * v2x + v1y * v2y
            return dot_product
        
        self.register_loss(f"right_{points[0].val}_{points[1].val}_{points[2].val}",
                      perpendicular_constraint,
                      weight=100.0)
    
        return [point_1, point_2, point_3]
    
    def sample_square(self, points, corner_point): # hình vuông
        assert len(points) == 4
        
        point_1 = self.sample_uniform(points[0])
        point_2 = self.sample_uniform(points[1])
        point_3 = self.sample_uniform(points[2])
        point_4 = self.sample_uniform(points[3])
        
        pts = [point_1, point_2, point_3, point_4]
        
        # constraint 1: 4 canh bang nhau AB = BC = CD = DA
        def equal_sides_constraint():
            d01 = self.dist(pts[0], pts[1]) # AB
            d12 = self.dist(pts[1], pts[2]) # BC
            d23 = self.dist(pts[2], pts[3]) # CD
            d30 = self.dist(pts[3], pts[0]) # DA
            
            avg=(d01 + d12 + d23 + d30) / 4.0
            return (d01 - avg)**2 + (d12 - avg)**2 + (d23 - avg)**2 + (d30 - avg)**2
        
        # constraint 2: 4 goc vuong nhau
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

        self.register_loss(f"square_angle_A", angle_at_A, weight=100.0)
        self.register_loss(f"square_angle_B", angle_at_B, weight=100.0)
        self.register_loss(f"square_angle_C", angle_at_C, weight=100.0)
        self.register_loss(f"square_angle_D", angle_at_D, weight=100.0) 
        
        self.register_ndg(f"square_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                     lambda a=pts[0], b=pts[1], c=pts[2]: self.collinear(a, b, c),
                     weight=1.0)
        return [point_1, point_2, point_3, point_4]
    
    
    
    def sample_rectangle(self, points, corner_point): # hình chữ nhật
        assert len(points) == 4
        
        point_1 = self.sample_uniform(points[0])
        point_2 = self.sample_uniform(points[1])
        point_3 = self.sample_uniform(points[2])
        point_4 = self.sample_uniform(points[3])
        
        pts = [point_1, point_2, point_3, point_4]
        
        # constraint 1: canh doi nhau bang nhau AB = CD, BC = DA
        def opposite_sides_constraint():
            d01 = self.dist(pts[0], pts[1]) # AB
            d12 = self.dist(pts[1], pts[2]) # BC
            d23 = self.dist(pts[2], pts[3]) # CD
            d30 = self.dist(pts[3], pts[0]) # DA
            return (d01 - d23)**2 + (d12 - d30)**2
        
        self.register_loss(
            f"rect_opposite_sides_{points[0].val}_{points[1].val}_{points[2].val}_{points[3].val}",
            opposite_sides_constraint,
            weight=10.0
        )
        
        # dieu chinh do dai ti le canh hinh chu nhat
        def aspect_ratio_constraint():
            d01 = self.dist(pts[0], pts[1])  # AB 
            d12 = self.dist(pts[1], pts[2])  # BC 
            
            # Đảm bảo BC > AB * 1.5
            ratio = d12 / (d01 + 1e-8)
            return torch.relu(1.5 - ratio)  # penalty nếu ratio < 1.5
        self.register_loss(f"rect_aspect_ratio", aspect_ratio_constraint, weight=5.0)

        # constraint 2: 4 goc vuong nhau
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

        self.register_loss(f"rect_angle_A", angle_at_A, weight=100.0)
        self.register_loss(f"rect_angle_B", angle_at_B, weight=100.0)
        self.register_loss(f"rect_angle_C", angle_at_C, weight=100.0)
        self.register_loss(f"rect_angle_D", angle_at_D, weight=100.0) 
        
        self.register_ndg(f"rect_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                     lambda a=pts[0], b=pts[1], c=pts[2]: self.collinear(a, b, c),
                     weight=1.0)
        
        return [point_1, point_2, point_3, point_4]
    

    
    def sample_rhombus (self, points, corner_point): # hình thoi
        assert len(points) == 4

        point_1 = self.sample_uniform(points[0])
        point_2 = self.sample_uniform(points[1])
        point_3 = self.sample_uniform(points[2])
        point_4 = self.sample_uniform(points[3])
        pts = [point_1, point_2, point_3, point_4]
        
        # constraint 1: 4 canh bang nhau AB = BC = CD = DA
        def equal_sides_constraint():
            d01 = self.dist(pts[0], pts[1]) # AB
            d12 = self.dist(pts[1], pts[2]) # BC
            d23 = self.dist(pts[2], pts[3]) # CD
            d30 = self.dist(pts[3], pts[0]) # DA
            
            avg=(d01 + d12 + d23 + d30) / 4.0
            return (d01 - avg)**2 + (d12 - avg)**2 + (d23 - avg)**2 + (d30 - avg)**2
        self.register_loss(f"rhombus_sides_{points[0].val}_{points[1].val}_{points[2].val}_{points[3].val}",
                      equal_sides_constraint,
                      weight=10.0)  
        
        # constraint 2: 2 duong cheo vuong goc
        def diagonals_perpendicular():
            v_ac_x = pts[2].x - pts[0].x # AC
            v_ac_y = pts[2].y - pts[0].y
            v_bd_x = pts[3].x - pts[1].x # BD
            v_bd_y = pts[3].y - pts[1].y
            return v_ac_x * v_bd_x + v_ac_y * v_bd_y
        
        self.register_loss(f"rhombus_diagonals_{points[0].val}_{points[1].val}_{points[2].val}_{points[3].val}",
                      diagonals_perpendicular,
                      weight=100.0)      
        self.register_ndg(f"rhombus_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                     lambda a=pts[0], b=pts[1], c=pts[2]: self.collinear(a, b, c),
                     weight=1.0)
        self.quadrilateral_type = "rhombus"
        return [point_1, point_2, point_3, point_4]
    
    
    
    def on_line(self, p: TorchPoint, line: LineNF):
        # line in normal form: n · p - r = 0
        return line.n.x * p.x + line.n.y * p.y - line.r
    
    def collinear(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint): # ktra 3 điểm thẳng hàng
        # dùng tích có hướng, (P2​−P1​)×(P3​−P1​)=0
        v1x = p2.x - p1.x
        v1y = p2.y - p1.y
        v2x = p3.x - p1.x
        v2y = p3.y - p1.y
        cross_product = v1x * v2y - v1y * v2x
        return cross_product
        
    
    def sample_isoceles_triangle(self, points: list, apex): # tam giác cân
        # apex --> cân tại đỉnh này
        assert len(points) == 3  # triangle defined by 3 points example (param (A B C) (iso-tri A))
        apex_idx = None
        for i, p in enumerate(points):
            if p.val == apex.val:
                apex_idx = i
                break
        if apex_idx is None:
            apex_idx = 0  # default to first point if apex not found
            
        point_1 = self.sample_uniform(points[0])
        point_2 = self.sample_uniform(points[1])
        point_3 = self.sample_uniform(points[2])\
            
        pts = [point_1, point_2, point_3]
        apex_point = pts[apex_idx]
        other_pts = [pts[i] for i in range(3) if i != apex_idx]

        # Constraint: equal distances from apex to other two points
        self.register_loss(f"iso_{points[0].val}_{points[1].val}_{points[2].val}",
                          lambda ap=apex_point, o0=other_pts[0], o1=other_pts[1]: self.dist(ap, o0) - self.dist(ap, o1),
                          weight=10.0)

        # Non-degeneracy
        self.register_ndg(f"tri_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda a=pts[0], b=pts[1], c=pts[2]: self.collinear(a, b, c), weight=1.0)

        # Lưu thông tin tam giác cân
        key = (points[0].val, points[1].val, points[2].val)
        self.iso_triangles[key] = apex_idx

        return [point_1, point_2, point_3]
        
    
    def process_parameter(self, instr):
        param_type = instr.param_type.lower()
        obj = instr.objects  
        args = instr.args
        
        if param_type == "triangle":
            self.sample_triangle(obj)
        elif param_type == "iso-tri":
            apex = args[0] if args else obj[0]
            self.sample_isoceles_triangle(obj, apex)   
        elif param_type == "right-tri":
            right_vertex = args[0] if args else obj[0]
            self.sample_right_triangle(obj, right_vertex)
        elif param_type == "square":
            corner = args[0] if args else None
            self.sample_square(obj, corner)
        elif param_type == "rectangle":
            corner = args[0] if args else None
            self.sample_rectangle(obj, corner)
        elif param_type == "rhombus":
            corner = args[0] if args else None
            self.sample_rhombus(obj, corner)
            
        elif param_type == "on-seg":
            self.paramerter_on_seg(obj[0], args)
        elif param_type == "on-line":
            self.parameter_on_line(obj[0], args)
        elif param_type == "coords":
            # Free point
            self.sample_uniform(obj[0])
        else:
            if self.verbose:
                logger.warning(f"Unsupported parameterization: {param_type}")
                
                
    def preprocess(self):
        for instr in self.instructions:
            self.process_parameter(instr)
            
    def regularize_points(self): # tránh điểm trùng nhau
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

    def verify_constraints(self):
        """Verify constraints after training"""
        with torch.no_grad():
            # Kiểm tra các tam giác vuông
            for key, loss_fn in self.loss_fns.items():
                if key.startswith("right_"):
                    # Tính lại constraint value (không phải loss)
                    # Loss = weight * (constraint)^2, ta cần constraint gốc
                    loss_val = loss_fn()
                    constraint_val = torch.sqrt(loss_val / 100.0)  # weight = 100
                    logger.info(f"{key} dot product: {constraint_val.item():.6f}")
            
    def train(self, epochs: int = 1000, lr: float = 0.01):
        if not self.has_loss:
            return 0.0

        optimizer = optim.Adam(self.trainable_val, lr=lr)

        if self.verbose:
            logger.info(f"Optimization ({epochs} Epochs")
        for i in range(epochs):
            optimizer.zero_grad()

            # Compute losses fresh at each iteration
            self.losses = {key: fn() for key, fn in self.loss_fns.items()}
            total_loss = sum(self.losses.values())

            total_loss.backward()

            optimizer.step()

            if self.verbose and i % 100 == 0:
                logger.info(f"Iteration {i:4d}: Loss = {total_loss.item():.6f}")

            # Early stopping
            if total_loss.item() < 1e-6:
                if self.verbose >= 0:
                    logger.info(f"Converged at iteration {i} with loss {total_loss.item():.6f}")
                break

        final_loss = total_loss.item()

        if self.verbose:
            logger.info(f"Final loss {final_loss:.6f}")
            self.log_losses()
            self.verify_constraints()

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

        # Try to detect triangles (3 consecutive points)
        point_names = list(self.name2pt.keys())
        n = len(point_names)
        
        if n == 4:
            # Hình vuông hoặc tứ giác
            p1 = diagram.points[point_names[0]]
            p2 = diagram.points[point_names[1]]
            p3 = diagram.points[point_names[2]]
            p4 = diagram.points[point_names[3]]
            
            # draw_diagonals = hasattr(self, 'quadrilateral_type') and self.quadrilateral_type == "rhombus"
            # diagram.add_quadrilateral(p1, p2, p3, p4, draw_diagonals=draw_diagonals)
            diagram.add_quadrilateral(p1, p2, p3, p4)
            
        elif n >= 3:
            for i in range(0, len(point_names) - 2, 3):
                p1_name = point_names[i]
                p2_name = point_names[i + 1]
                p3_name = point_names[i + 2]

                p1 = diagram.points[p1_name]
                p2 = diagram.points[p2_name]
                p3 = diagram.points[p3_name]

                # Kiểm tra nếu là tam giác cân
                key = (p1_name, p2_name, p3_name)
                equal_sides = None
                if key in self.iso_triangles:
                    apex_idx = self.iso_triangles[key]
                    # apex_idx là đỉnh, 2 cạnh bằng nhau là từ apex đến 2 đỉnh còn lại
                    others = [j for j in range(3) if j != apex_idx]
                    equal_sides = [(apex_idx, others[0]), (apex_idx, others[1])]

                diagram.add_triangle(p1, p2, p3, equal_sides)

        return diagram