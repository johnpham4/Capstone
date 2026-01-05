"""Simplified instruction reader for triangles and lines only"""
import math
import numpy as np

from instruction import Assert, AssertNDG, Eval, Sample, Parameterize, Compute
from constraint import Constraint
from parse import parse_sexprs
from primitives import Point, Line, Num
from util import Root, is_number, FuncInfo

RESERVED_NAMES = ["pi"]


class InstructionReader:
    def __init__(self, problem_lines):
        self.points = list()
        self.lines = list()
        self.instructions = list()
        self.problem_lines = problem_lines

        self.unnamed_points = list()
        self.unnamed_lines = list()
        self.segments = list()
        self.seg_colors = list()

        cmds = parse_sexprs(self.problem_lines)
        for cmd in cmds:
            try:
                self.process_command(cmd)
            except Exception as e:
                raise RuntimeError(f"Invalid command: {cmd}, Error: {e}")

    def register_pt(self, p):
        if p in self.points:
            raise RuntimeError(f"Same point declared twice: {p}")
        if not (isinstance(p, Point) and isinstance(p.val, str)):
            raise RuntimeError(f"Invalid point: {p}")
        if p.val.lower() in RESERVED_NAMES:
            raise RuntimeError(f"Reserved name: {p}")
        self.points.append(p)

    def register_line(self, l):
        if l in self.lines:
            raise RuntimeError(f"Same line declared twice: {l}")
        if not (isinstance(l, Line) and isinstance(l.val, str)):
            raise RuntimeError(f"Invalid line name: {l.val}")
        if l.val.lower() in RESERVED_NAMES:
            raise RuntimeError(f"Reserved name: {l}")
        self.lines.append(l)

    def process_command(self, cmd):
        if not isinstance(cmd[0], str):
            raise RuntimeError("Command must be a string")
        head = cmd[0].lower()
        if head == "assert":
            self.add(cmd)
        elif head == "define":
            self.compute(cmd)
        elif head == "eval":
            self.eval_cons(cmd)
        elif head == "param":
            if isinstance(cmd[1], str):
                self.param(cmd)
            elif isinstance(cmd[1], tuple):
                self.process_param_special(cmd)
            else:
                raise RuntimeError("Invalid param input type")
        else:
            raise NotImplementedError(f"Command not supported: {head}")

    def process_param_special(self, cmd):
        """Handle triangle sampling"""
        assert len(cmd) == 3
        ps = [Point(p) for p in cmd[1]]
        for p in ps:
            self.register_pt(p)

        param_method = cmd[2]

        if isinstance(param_method, str):
            p_method = param_method.lower()
            assert p_method in ["triangle", "acute-tri", "equi-tri"]
            instr = Sample(ps, p_method)
            self.instructions.append(instr)
        elif isinstance(param_method, tuple):
            assert len(param_method) == 2
            head, arg = param_method
            head = head.lower()
            assert head in ["right-tri", "iso-tri", "acute-iso-tri"]
            assert isinstance(arg, str) and Point(arg) in ps
            special_p = Point(arg)
            instr = Sample(ps, head, (special_p,))
            self.instructions.append(instr)
        else:
            raise RuntimeError("Invalid joint param method")

        # Add segments for triangle
        n_gon_color = np.random.rand(3)
        for i in range(len(ps)):
            self.segments.append((ps[i], ps[(i+1) % len(ps)]))
            self.seg_colors.append(n_gon_color)

    def compute(self, cmd):
        """Define new geometric objects"""
        assert len(cmd) == 4
        obj_name = cmd[1]
        assert isinstance(obj_name, str)
        obj_type = cmd[2].lower()
        assert obj_type in ["point", "line"]

        if obj_type == "point":
            p = Point(obj_name)
            self.register_pt(p)
            computation = self.process_point(cmd[3], unnamed=False)
            assert not isinstance(computation.val, str)
            c_instr = Compute(p, computation)
            self.instructions.append(c_instr)
        elif obj_type == "line":
            l = Line(obj_name)
            self.register_line(l)
            computation = self.process_line(cmd[3], unnamed=False)
            assert not isinstance(computation.val, str)
            c_instr = Compute(l, computation)
            self.instructions.append(c_instr)
        else:
            raise RuntimeError("Invalid define type")

    def add(self, cmd):
        """Add assertion constraint"""
        assert len(cmd) == 2
        negate, pred, args = self.process_constraint(cmd[1])
        instr_cons = Constraint(pred, args, False)
        if negate:
            self.instructions.append(AssertNDG(instr_cons))
        else:
            self.instructions.append(Assert(instr_cons))

    def eval_cons(self, cmd):
        """Evaluate constraint as goal"""
        assert len(cmd) == 2
        negate, pred, args = self.process_constraint(cmd[1])
        instr_cons = Constraint(pred, args, negate)
        self.instructions.append(Eval(instr_cons))

    def param(self, cmd):
        """Parameterize object"""
        assert len(cmd) == 3 or len(cmd) == 4
        obj_type = cmd[2].lower()
        assert obj_type in ["point", "line"]

        if obj_type == "line":
            l = Line(cmd[1])
            self.register_line(l)
            param_method = "line"
            if len(cmd) == 4:
                param_method = cmd[3]
            pred, args = self.process_param_line(param_method)
            p_instr = Parameterize(l, (pred, args))
            self.instructions.append(p_instr)
        else:
            p = Point(cmd[1])
            self.register_pt(p)
            param_method = "coords"
            if len(cmd) == 4:
                param_method = cmd[3]
            pred, args = self.process_param_point(param_method)
            p_instr = Parameterize(p, (pred, args))
            self.instructions.append(p_instr)

    def process_param_line(self, param):
        """Process line parameterization"""
        if isinstance(param, str) and param.lower() == "line":
            return "line", None
        pred = param[0].lower()
        args = param[1:]
        args = [self.process_term(t) for t in args]
        if pred == "through":
            assert len(args) == 1
            assert isinstance(args[0], Point)
            pred = "through-l"
        else:
            raise NotImplementedError(f"Unrecognized param {param}")
        return pred, args

    def process_param_point(self, param):
        """Process point parameterization"""
        if isinstance(param, str) and param.lower() == "coords":
            return "coords", None
        pred = param[0].lower()
        args = param[1:]
        args = [self.process_term(t) for t in args]
        if pred == "on-seg":
            assert len(args) == 2
            assert all([isinstance(t, Point) for t in args])
        elif pred == "on-line":
            assert len(args) == 1
            assert all([isinstance(t, Line) for t in args])
        elif pred in ["on-ray", "on-ray-opp"]:
            assert len(args) == 2
            assert all([isinstance(t, Point) for t in args])
        else:
            raise NotImplementedError(f"Unrecognized param {param}")
        return pred, args

    def process_constraint(self, constraint):
        """Parse constraint from S-expression"""
        assert isinstance(constraint, tuple)
        negate = (isinstance(constraint[0], str) and constraint[0].lower() == "not")
        if negate:
            constraint = constraint[1]

        pred = constraint[0].lower()
        args = constraint[1:]
        args = [self.process_term(t) for t in args]

        # Validate common constraints
        if pred == "coll":
            assert len(args) == 3
            assert all([isinstance(t, Point) for t in args])
        elif pred == "cong":
            assert len(args) == 4
            assert all([isinstance(t, Point) for t in args])
        elif pred == "eq" or pred == "=":
            assert len(args) == 2
            if all([isinstance(t, Num) for t in args]):
                pred = "eq-n"
            elif all([isinstance(t, Point) for t in args]):
                pred = "eq-p"
            elif all([isinstance(t, Line) for t in args]):
                pred = "eq-l"
        elif pred in ["gt", ">", "lt", "<", "gte", ">=", "lte", "<="]:
            assert len(args) == 2
            assert all([isinstance(t, Num) for t in args])
            if pred == ">": pred = "gt"
            elif pred == "<": pred = "lt"
            elif pred == ">=": pred = "gte"
            elif pred == "<=": pred = "lte"
        elif pred == "midp":
            assert len(args) == 3
            assert all([isinstance(t, Point) for t in args])
        elif pred in ["on-seg", "on-ray"]:
            assert len(args) == 3
            assert all([isinstance(t, Point) for t in args])
        elif pred == "on-line":
            assert len(args) == 2
            assert isinstance(args[0], Point) and isinstance(args[1], Line)
        elif pred in ["para", "perp"]:
            assert len(args) == 2
            assert all([isinstance(t, Line) for t in args])
        elif pred in ["right", "right-tri"]:
            assert len(args) == 3
            assert all([isinstance(t, Point) for t in args])
        elif pred == "same-side" or pred == "opp-sides":
            assert len(args) == 3
            assert isinstance(args[0], Point) and isinstance(args[1], Point)
            assert isinstance(args[2], Line)
        elif pred == "centroid" or pred == "incenter" or pred == "circumcenter" or pred == "orthocenter":
            assert len(args) == 4
            assert all([isinstance(t, Point) for t in args])
        elif pred == "i-bisector":
            assert len(args) == 4
            assert all([isinstance(t, Point) for t in args])
        elif pred == "inter-ll":
            assert len(args) == 5
            assert all([isinstance(t, Point) for t in args])
        else:
            raise NotImplementedError(f"Unsupported predicate {pred}")

        return negate, pred, args

    def process_term(self, term):
        """Process a term (point, line, or number)"""
        try:
            return self.process_point(term)
        except:
            try:
                return self.process_line(term)
            except:
                try:
                    return self.process_number(term)
                except:
                    raise RuntimeError(f"Term {term} not a point/line/number")

    def process_point(self, p_info, unnamed=True):
        """Process point definition"""
        if isinstance(p_info, str) and not is_number(p_info):
            assert Point(p_info) in self.points
            return Point(p_info)
        if not isinstance(p_info, tuple):
            raise NotImplementedError("p_info must be tuple or string")

        p_pred = p_info[0].lower()
        p_args = p_info[1:]
        p_val = None

        if p_pred == "inter-ll":
            assert len(p_args) == 2
            l1 = self.process_line(p_args[0])
            l2 = self.process_line(p_args[1])
            p_val = FuncInfo(p_pred, (l1, l2))
        elif p_pred in ["incenter", "centroid", "circumcenter", "orthocenter"]:
            assert len(p_args) == 3
            ps = [self.process_point(p) for p in p_args]
            p_val = FuncInfo(p_pred, tuple(ps))
        elif p_pred in ["midp", "midp-from"]:
            assert len(p_args) == 2
            ps = [self.process_point(p) for p in p_args]
            p_val = FuncInfo(p_pred, tuple(ps))
        elif p_pred == "foot":
            assert len(p_args) == 2
            p = self.process_point(p_args[0])
            l = self.process_line(p_args[1])
            p_val = FuncInfo(p_pred, (p, l))
        else:
            raise NotImplementedError(f"Unsupported point pred {p_pred}")

        if unnamed:
            self.unnamed_points.append(Point(p_val))
        return Point(p_val)

    def process_line(self, l_info, unnamed=True):
        """Process line definition"""
        if isinstance(l_info, str):
            assert Line(l_info) in self.lines
            return Line(l_info)
        if not isinstance(l_info, tuple):
            raise NotImplementedError("l_info must be tuple or string")

        l_pred = l_info[0].lower()
        l_args = l_info[1:]
        l_val = None

        if l_pred == "connecting":
            assert len(l_args) == 2
            ps = [self.process_point(p) for p in l_args]
            l_val = FuncInfo(l_pred, tuple(ps))
        elif l_pred == "para-at":
            assert len(l_args) == 2
            p = self.process_point(l_args[0])
            l = self.process_line(l_args[1])
            l_val = FuncInfo(l_pred, (p, l))
        elif l_pred == "perp-at":
            assert len(l_args) == 2
            p = self.process_point(l_args[0])
            l = self.process_line(l_args[1])
            l_val = FuncInfo(l_pred, (p, l))
        elif l_pred == "mediator" or l_pred == "perp-bis":
            assert len(l_args) == 2
            ps = [self.process_point(p) for p in l_args]
            l_val = FuncInfo("mediator", tuple(ps))
        elif l_pred == "i-bisector":
            assert len(l_args) == 3
            ps = [self.process_point(p) for p in l_args]
            l_val = FuncInfo(l_pred, tuple(ps))
        else:
            raise NotImplementedError(f"Unsupported line pred {l_pred}")

        if unnamed:
            self.unnamed_lines.append(Line(l_val))
        return Line(l_val)

    def process_number(self, n_info):
        """Process number value"""
        if isinstance(n_info, str) and is_number(n_info):
            return Num(float(n_info))
        if isinstance(n_info, str) and n_info.lower() == "pi":
            return Num(math.pi)
        if not isinstance(n_info, tuple):
            raise NotImplementedError("n_info must be tuple or string")

        n_pred = n_info[0].lower()
        n_args = n_info[1:]

        if n_pred == "dist":
            assert len(n_args) == 2
            ps = [self.process_point(p) for p in n_args]
            return Num(FuncInfo(n_pred, tuple(ps)))
        elif n_pred == "uangle" or n_pred == "angle":
            assert len(n_args) == 3
            ps = [self.process_point(p) for p in n_args]
            return Num(FuncInfo("uangle", tuple(ps)))
        elif n_pred == "area":
            assert len(n_args) == 3
            ps = [self.process_point(p) for p in n_args]
            return Num(FuncInfo(n_pred, tuple(ps)))
        elif n_pred in ["div", "add", "mul", "sub", "pow"]:
            assert len(n_args) == 2
            ns = [self.process_number(n) for n in n_args]
            return Num(FuncInfo(n_pred, tuple(ns)))
        elif n_pred in ["neg", "sqrt"]:
            assert len(n_args) == 1
            n = self.process_number(n_args[0])
            return Num(FuncInfo(n_pred, (n,)))
        else:
            raise NotImplementedError(f"Unsupported number pred {n_pred}")

    def process_rs(self, rs_info):
        """Process root selection"""
        if isinstance(rs_info, str):
            return Root("arbitrary", list())
        pred = rs_info[0].lower()
        args = rs_info[1:]
        return Root(pred, args)
