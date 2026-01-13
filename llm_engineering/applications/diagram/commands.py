from parser import Parser
from primitives import *
from instructions import *

class Command:
    def __init__(self, problem_lines: list[str]):
        self.points = list()
        self.lines = list()

        self.instructions = list()
        self.problem_lines = problem_lines

        self.unnamed_points = list()
        self.unnamed_lines = list()
        self.segments = list()
        self.seg_colors = list()

        cmds = Parser.parse_sexprs(self.problem_lines)
        for cmd in cmds:
            try:
                self.process_command(cmd)
            except:
                raise RuntimeError(f"Invalid command: {cmd}")


    def process_command(self, cmd: list[tuple]):
        if not isinstance(cmd[0], str):
            raise RuntimeError("Command must start with a string")

        head = cmd[0].lower()

        if head == "param":
            self.process_param(cmd)
        elif head == "assert":
            self.process_assert(cmd)
        else:
            raise NotImplementedError(f"Command not supported: {head}")

    def register_pt(self, p: Point):
        if p in self.points:
            raise RuntimeError(f"Same point declared twice: {p}")
        self.points.append(p)

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
                instr = Parameter(ps, param_method.lower())
                self.instructions.append(instr)
            elif isinstance(param_method, tuple):
                # Parameterization with argument like (iso-tri A)
                head = param_method[0].lower()
                arg = param_method[1]
                special_p = Point(arg)
                instr = Parameter(ps, head, (special_p,))
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
                        instr = Parameter([p], pred, tuple(args))
                        self.instructions.append(instr)

