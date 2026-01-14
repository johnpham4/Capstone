import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple

from instructions import Parameter
from primitives import *

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
        self.goals = {}  # Goal constraints to achieve
        self.iso_triangles = {}  # Lưu thông tin tam giác cân: key -> apex_idx

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

    def mkvar(self, name, lo=-1.0, hi=1.0):
        """Create a trainable variable"""
        val = torch.empty(1, dtype=torch.float64, device=self.device).uniform_(lo, hi)
        param = nn.Parameter(val)
        self.trainable_vars.append(param)
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

    def on_line(self, p: TorchPoint, line: LineNF):
        # line in normal form: n · p - r = 0
        return line.n.x * p.x + line.n.y * p.y - line.r

    def collinear(self, p1: TorchPoint, p2: TorchPoint, p3: TorchPoint):
        # Use cross product: (p2-p1) × (p3-p1) = 0
        v1x = p2.x - p1.x
        v1y = p2.y - p1.y
        v2x = p3.x - p1.x
        v2y = p3.y - p1.y
        return v1x * v2y - v1y * v2x

    def register_pt(self, p: TorchPoint, P, save_name=True):
        if save_name:
            assert p.val not in self.name2pt
            self.name2pt[p.val] = P

        self.all_points.append(P)
        return P

    def register_line(self, l, L):
        """Register a line with its name"""
        assert l.val not in self.name2line
        self.name2line[l.val] = L
        return L

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


    def sample_uniform(self, p, lo=-1.0, hi=1.0, save_name=True):
        """Sample a point uniformly in a box"""
        x = self.mkvar(f"{p.val}_x", lo, hi)
        y = self.mkvar(f"{p.val}_y", lo, hi)
        P = self.get_point(x, y)
        return self.register_pt(p, P, save_name)

    def sample_triangle(self, points: list):
        assert len(points) == 3

        # Create points with learnable coordinates
        p1 = self.sample_uniform(points[0])
        p2 = self.sample_uniform(points[1])
        p3 = self.sample_uniform(points[2])

        # Add non-degeneracy constraint (points should not be collinear)
        self.register_ndg(f"tri_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda a=p1, b=p2, c=p3: self.collinear(a, b, c), weight=1.0)

        return [p1, p2, p3]

    def sample_isoceles_triangle(self, points: list, apex):
        assert len(points) == 3

        apex_idx = None
        for i, p in enumerate(points):
            if p.val == apex.val:
                apex_idx = i
                break

        if apex_idx is None:
            apex_idx = 0

        # Create points
        p1 = self.sample_uniform(points[0])
        p2 = self.sample_uniform(points[1])
        p3 = self.sample_uniform(points[2])

        pts = [p1, p2, p3]
        apex_pt = pts[apex_idx]
        other_pts = [pts[i] for i in range(3) if i != apex_idx]

        # Constraint: equal distances from apex to other two points
        self.register_loss(f"iso_{points[0].val}_{points[1].val}_{points[2].val}",
                          lambda ap=apex_pt, o0=other_pts[0], o1=other_pts[1]: self.dist(ap, o0) - self.dist(ap, o1),
                          weight=10.0)

        # Non-degeneracy
        self.register_ndg(f"tri_ndg_{points[0].val}_{points[1].val}_{points[2].val}",
                         lambda a=p1, b=p2, c=p3: self.collinear(a, b, c), weight=1.0)

        # Lưu thông tin tam giác cân
        key = (points[0].val, points[1].val, points[2].val)
        self.iso_triangles[key] = apex_idx

        return [p1, p2, p3]

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

    def parameter_on_line(self, p, line_points):
        assert len(line_points) == 2

        p1 = self.lookup_pt(line_points[0])
        p2 = self.lookup_pt(line_points[1])

        # Create the line
        line = self.pp2lnf(p1, p2)

        # Create a free point
        P = self.sample_uniform(p, save_name=False)

        # Constrain it to be on the line
        self.register_loss(f"on_line_{p.val}",
                          lambda pt=P, ln=line: self.on_line(pt, ln), weight=10.0)

        return self.register_pt(p, P)

    def process_instruction(self, instr):
        if isinstance(instr, Parameter):
            self.process_parameter(instr)
        # elif isinstance(instr, Assert):
        #     self.process_assert(instr)

    def process_parameter(self, instr):
        param_type = instr.param_type
        objects = instr.objects
        args = instr.args

        if param_type == "triangle":
            self.sample_triangle(objects)
        elif param_type == "iso-tri":
            # Isosceles triangle with apex specified in args
            apex = args[0] if args else objects[0]
            self.sample_isoceles_triangle(objects, apex)
        elif param_type == "on-seg":
            self.paramerter_on_seg(objects[0], args)
        elif param_type == "on-line":
            self.parameter_on_line(objects[0], args)
        elif param_type == "coords":
            # Free point
            self.sample_uniform(objects[0])
        else:
            if self.verbosity:
                logger.warning(f"Unsupported parameterization: {param_type}")


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
            logger.info(f"Optimization ({epochs} Epochs")

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
        from diagram import Diagram

        diagram = Diagram()

        # Convert points
        for name, pt in self.name2pt.items():
            x = pt.x.detach().cpu().item()
            y = pt.y.detach().cpu().item()
            geo_pt = GeometricPoint(x, y, name)
            diagram.add_point(name, geo_pt)

        # Try to detect triangles (3 consecutive points)
        point_names = list(self.name2pt.keys())
        if len(point_names) >= 3:
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