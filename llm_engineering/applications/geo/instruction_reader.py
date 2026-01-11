"""
Instruction reader for parsing geometry problems
"""

from .primitives import Point, Line
from .constraint import Constraint
from .instruction import Parameterize, Assert
from .parse import parse_sexprs
from .util import is_number


class InstructionReader:
    """Reads and parses geometry problem instructions"""

    def __init__(self, problem_lines):
        self.points = []
        self.lines = []
        self.instructions = []
        self.problem_lines = problem_lines

        cmds = parse_sexprs(self.problem_lines)
        for cmd in cmds:
            try:
                self.process_command(cmd)
            except Exception as e:
                raise RuntimeError(f"Invalid command: {cmd}, Error: {e}")

    def register_pt(self, p):
        """Register a point"""
        if p in self.points:
            raise RuntimeError(f"Same point declared twice: {p}")
        if not (isinstance(p, Point) and isinstance(p.val, str)):
            raise RuntimeError(f"Invalid point: {p}")
        self.points.append(p)

    def process_command(self, cmd):
        """Process a single command"""
        if not isinstance(cmd[0], str):
            raise RuntimeError("Command must start with a string")

        head = cmd[0].lower()

        if head == "param":
            self.process_param(cmd)
        elif head == "assert":
            self.process_assert(cmd)
        else:
            raise NotImplementedError(f"Command not supported: {head}")

    def process_param(self, cmd):
        """Process parameterization command"""
        # Format: (param (A B C) (iso-tri A))
        # or: (param D point (on-seg A B))

        if isinstance(cmd[1], tuple):
            # Multi-point parameterization like (param (A B C) (iso-tri A))
            ps = [Point(p) for p in cmd[1]]
            for p in ps:
                self.register_pt(p)

            param_method = cmd[2]

            if isinstance(param_method, str):
                # Simple parameterization like "triangle"
                instr = Parameterize(ps, param_method.lower())
                self.instructions.append(instr)
            elif isinstance(param_method, tuple):
                # Parameterization with argument like (iso-tri A)
                head = param_method[0].lower()
                arg = param_method[1]
                special_p = Point(arg)
                instr = Parameterize(ps, head, (special_p,))
                self.instructions.append(instr)
        else:
            # Single point parameterization like (param D point (on-seg A B))
            obj_name = cmd[1]
            obj_type = cmd[2].lower()

            if obj_type == "point":
                p = Point(obj_name)
                self.register_pt(p)

                if len(cmd) > 3:
                    # Has constraint like (on-seg A B)
                    constraint_info = cmd[3]
                    if isinstance(constraint_info, tuple):
                        pred = constraint_info[0].lower()
                        args = [Point(a) for a in constraint_info[1:]]
                        instr = Parameterize([p], pred, tuple(args))
                        self.instructions.append(instr)

    def process_assert(self, cmd):
        """Process assertion command"""
        # Format: (assert (constraint args...))
        constraint_data = cmd[1]

        if isinstance(constraint_data, tuple):
            pred = constraint_data[0].lower()
            args = [Point(a) if isinstance(a, str) and not is_number(a) else a
                   for a in constraint_data[1:]]

            constraint = Constraint(pred, args, negate=False)
            self.instructions.append(Assert(constraint))
