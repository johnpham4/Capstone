from typing import List, Tuple, Any

from llm_engineering.applications.diagram.services.dsl_parser import DSLParser
from llm_engineering.domains.geometry import Point
from llm_engineering.domains.geometry.instructions import Parameter
from llm_engineering.domains.geometry.types import DiagramType, TriangleType


class DiagramBuilder:
    def __init__(self, problem_lines: List[str]):
        self.points: List[Point] = []
        self.instructions: List[Any] = []
        self.problem_lines = problem_lines

        cmds = DSLParser.parse_sexprs(self.problem_lines)
        for cmd in cmds:
            try:
                self.process_command(cmd)
            except:
                raise RuntimeError(f"Invalid command: {cmd}")

    def process_command(self, cmd: Tuple):
        if not isinstance(cmd[0], str):
            raise RuntimeError("Command must start with a string")

        head = cmd[0].lower()

        if head == "triangle":
            self.process_triangle(cmd)
        elif head == "define":
            self.process_define(cmd)
        elif head == "circle":
            self.process_circle(cmd)
        elif head == "segment":
            self.process_segment(cmd)
        elif head == "line":
            self.process_line(cmd)
        else:
            raise NotImplementedError(f"Command not supported: {head}")

    def register_pt(self, p: Point):
        if p in self.points:
            raise RuntimeError(f"Same point declared twice: {p}")
        self.points.append(p)

    def process_triangle(self, cmd):
        ps = [Point(p) for p in cmd[1]]
        for p in ps:
            self.register_pt(p)

        if len(cmd) == 2:
            instr = Parameter(DiagramType.TRIANGLE, ps, None)
            self.instructions.append(instr)
            return

        param_method = cmd[2]

        if len(param_method) == 1:
            type_str = param_method[0].upper().replace('-', '_')

            try:
                head = TriangleType[type_str]
                instr = Parameter(DiagramType.TRIANGLE, ps, head)
            except KeyError:
                instr = Parameter(DiagramType.TRIANGLE, ps, param_method[0].lower())

            self.instructions.append(instr)
            return

        # Two parameters: type + special point
        type_str = param_method[0].upper().replace('-', '_')
        special_p = Point(param_method[1])

        try:
            head = TriangleType[type_str]
            instr = Parameter(DiagramType.TRIANGLE, ps, head, (special_p,))
        except KeyError:
            instr = Parameter(DiagramType.TRIANGLE, ps, param_method[0].lower(), (special_p,))

        self.instructions.append(instr)

    def process_define(self, cmd):
        """Process: (define G point (centroid A B C))"""
        if len(cmd) < 4:
            raise RuntimeError(f"Invalid define command: {cmd}")

        point_name = cmd[1]
        obj_type = cmd[2]
        construction = cmd[3]

        if obj_type != "point":
            raise NotImplementedError(f"Only 'point' type supported, got: {obj_type}")

        if not isinstance(construction, tuple):
            raise RuntimeError(f"Construction must be a tuple: {construction}")

        construction_type = construction[0].lower()
        construction_args = construction[1:]

        # Handle special cases where args contain nested structures
        # e.g., (projection A (segment B C))
        processed_args = []
        for arg in construction_args:
            if isinstance(arg, tuple):
                # Flatten nested structures like (segment B C) -> [B, C]
                processed_args.extend(arg[1:])
            else:
                processed_args.append(arg)

        # Create Parameter instruction for geometric construction
        instr = Parameter(
            diagram_type=DiagramType.POINT,
            objects=[Point(point_name)],
            param_type=construction_type,
            args=tuple(Point(p) for p in processed_args)
        )
        self.instructions.append(instr)

    def process_circle(self, cmd):
        """Process: (circle I (incircle A B C))"""
        if len(cmd) < 3:
            raise RuntimeError(f"Invalid circle command: {cmd}")

        center_name = cmd[1]
        construction = cmd[2]

        if not isinstance(construction, tuple):
            raise RuntimeError(f"Circle construction must be a tuple: {construction}")

        construction_type = construction[0].lower()
        construction_args = construction[1:]

        # Create Parameter instruction for circle
        instr = Parameter(
            diagram_type=DiagramType.CIRCLE,
            objects=[Point(center_name)],
            param_type=construction_type,
            args=tuple(Point(p) for p in construction_args)
        )
        self.instructions.append(instr)

    def process_segment(self, cmd):
        """Process: (segment A M)"""
        if len(cmd) != 3:
            raise RuntimeError(f"Segment requires 2 points: {cmd}")

        p1 = Point(cmd[1])
        p2 = Point(cmd[2])

        instr = Parameter(
            diagram_type=DiagramType.SEGMENT,
            objects=[p1, p2],
            param_type="segment",
            args=()
        )
        self.instructions.append(instr)

    def process_line(self, cmd):
        """Process: (line A B)"""
        if len(cmd) != 3:
            raise RuntimeError(f"Line requires 2 points: {cmd}")

        p1 = Point(cmd[1])
        p2 = Point(cmd[2])

        instr = Parameter(
            diagram_type=DiagramType.LINE,
            objects=[p1, p2],
            param_type="line",
            args=()
        )
        self.instructions.append(instr)
