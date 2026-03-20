from typing import List, Tuple, Any

from src.services.diagram.dsl_parser import DSLParser
from src.services.diagram.models import Point
from src.services.diagram.models.instructions import Parameter
from src.services.diagram.models.types import DiagramType, QuadrilateralType, TriangleType


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
        elif head == "square":
            self.process_square(cmd)
        elif head == "define":
            self.process_define(cmd)
        elif head == "circle":
            self.process_circle(cmd)
        elif head == "segment":
            self.process_segment(cmd)
        elif head == "line":
            self.process_line(cmd)
        elif head == "parallel":
            self.process_parallel(cmd)
        elif head == "perpendicular":
            self.process_perpendicular(cmd)
        elif head == "angle-equal":
            self.process_angle_equal(cmd)
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

        # Two or more parameters: type + arguments
        type_str = param_method[0].upper().replace('-', '_')

        # Handle different parameter types
        param_type_lower = param_method[0].lower()
        if param_type_lower in ['equal_angles', 'equal-angles']:
            # For equal_angles, args are indices (integers), not Points
            args = tuple(param_method[1:])
        else:
            # For other types (isosceles, right, etc.), args are Points
            args = tuple(Point(p) for p in param_method[1:])

        try:
            head = TriangleType[type_str]
            instr = Parameter(DiagramType.TRIANGLE, ps, head, args)
        except KeyError:
            instr = Parameter(DiagramType.TRIANGLE, ps, param_type_lower, args)

        self.instructions.append(instr)

    def process_square(self, cmd):
        """Process square command: (square (A B C D))"""
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid square command: {cmd}")

        ps = [Point(p) for p in cmd[1]]

        if len(ps) != 4:
            raise ValueError(f"Square must have 4 vertices, got {len(ps)}")

        # Register all 4 points
        for p in ps:
            self.register_pt(p)

        instr = Parameter(
            DiagramType.QUADRILATERAL,
            ps,
            QuadrilateralType.SQUARE,
            (ps[0],)  # Corner point (default first vertex)
        )

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

    def process_parallel(self, cmd):
        """Process: (parallel (segment B C) (segment D E))"""
        if len(cmd) != 3:
            raise RuntimeError(f"Parallel requires 2 segments: {cmd}")

        seg1 = cmd[1]
        seg2 = cmd[2]

        if not (isinstance(seg1, tuple) and isinstance(seg2, tuple)):
            raise RuntimeError(f"Parallel arguments must be segments: {cmd}")

        if seg1[0] != "segment" or seg2[0] != "segment":
            raise RuntimeError(f"Parallel requires segments: {cmd}")

        # Extract 4 points: B, C, D, E
        p1 = Point(seg1[1])
        p2 = Point(seg1[2])
        p3 = Point(seg2[1])
        p4 = Point(seg2[2])

        from src.services.diagram.models.instructions import Assertion
        instr = Assertion(
            constraint_type='parallel',
            objects=[p1, p2, p3, p4]
        )
        self.instructions.append(instr)

    def process_perpendicular(self, cmd):
        """Process: (perpendicular (segment A B) (segment C D))"""
        if len(cmd) != 3:
            raise RuntimeError(f"Perpendicular requires 2 segments: {cmd}")

        seg1 = cmd[1]
        seg2 = cmd[2]

        if not (isinstance(seg1, tuple) and isinstance(seg2, tuple)):
            raise RuntimeError(f"Perpendicular arguments must be segments: {cmd}")

        if seg1[0] != "segment" or seg2[0] != "segment":
            raise RuntimeError(f"Perpendicular requires segments: {cmd}")

        # Extract 4 points
        p1 = Point(seg1[1])
        p2 = Point(seg1[2])
        p3 = Point(seg2[1])
        p4 = Point(seg2[2])

        from src.services.diagram.models.instructions import Assertion
        instr = Assertion(
            constraint_type='perpendicular',
            objects=[p1, p2, p3, p4]
        )
        self.instructions.append(instr)

    def process_angle_equal(self, cmd):
        """Process: (angle-equal A B C D E F) -> ∠ABC = ∠DEF"""
        if len(cmd) != 7:
            raise RuntimeError(f"angle-equal requires 6 points: {cmd}")

        # Extract 6 points for two angles
        points = [Point(cmd[i]) for i in range(1, 7)]

        from src.services.diagram.models.instructions import Assertion
        instr = Assertion(
            constraint_type='angle_equal',
            objects=points
        )
        self.instructions.append(instr)
