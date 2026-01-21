from typing import List, Tuple, Any

from llm_engineering.applications.diagram.services.dsl_parser import DSLParser
from llm_engineering.domains.geometry import Point
from llm_engineering.domains.geometry.instructions import Parameter
from llm_engineering.domains.geometry.types import DiagramType, TriangleType, QuadrilateralType


class DiagramBuilder:
    def __init__(self, problem_lines: List[str]):
        self.points: List[Point] = []
        self.instructions: List[Any] = []
        self.problem_lines = problem_lines

        cmds = DSLParser.parse_sexprs(self.problem_lines)
        for cmd in cmds:
            try:
                self.process_command(cmd)
            except Exception as e:
                raise RuntimeError(f"Invalid command: {cmd}. Error: {e}")

    def process_command(self, cmd: Tuple):
        if not isinstance(cmd[0], str):
            raise RuntimeError("Command must start with a string")

        head = cmd[0].lower()

        if head == "triangle":
            self.process_triangle(cmd)
        elif head == "square":
            self.process_square(cmd)
        elif head == "rectangle":
            self.process_rectangle(cmd)
        elif head == "parallelogram":
            self.process_parallelogram(cmd)
        elif head == "trapezoid":
            self.process_trapezoid(cmd)
        elif head == "rhombus":  
            self.process_rhombus(cmd)
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
            pass
        else:
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

        type_str = param_method[0].upper().replace('-', '_')
        special_p = Point(param_method[1])

        try:
            head = TriangleType[type_str]
            instr = Parameter(DiagramType.TRIANGLE, ps, head, (special_p,))
        except KeyError:
            instr = Parameter(DiagramType.TRIANGLE, ps, param_method[0].lower(), (special_p,))

        self.instructions.append(instr)

    def process_square(self, cmd):
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid square command: {cmd}")
        
        points_data = cmd[1]
        if len(points_data) != 4:
            raise RuntimeError(f"Square requires exactly 4 points, got: {len(points_data)}")

        ps = [Point(p) for p in points_data]
        for p in ps:
            self.register_pt(p)

        instr = Parameter(DiagramType.QUADRILATERAL, ps, QuadrilateralType.SQUARE)
        self.instructions.append(instr)

    def process_rectangle(self, cmd):
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid rectangle command: {cmd}")
        
        points_data = cmd[1]
        if len(points_data) != 4:
            raise RuntimeError(f"Rectangle requires exactly 4 points, got: {len(points_data)}")

        ps = [Point(p) for p in points_data]
        for p in ps:
            self.register_pt(p)

        instr = Parameter(DiagramType.QUADRILATERAL, ps, QuadrilateralType.RECTANGLE)
        self.instructions.append(instr)

    def process_parallelogram(self, cmd):
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid parallelogram command: {cmd}")
        
        points_data = cmd[1]
        if len(points_data) != 4:
            raise RuntimeError(f"Parallelogram requires exactly 4 points, got: {len(points_data)}")

        ps = [Point(p) for p in points_data]
        for p in ps:
            self.register_pt(p)

        instr = Parameter(DiagramType.QUADRILATERAL, ps, QuadrilateralType.PARALLELOGRAM)
        self.instructions.append(instr)

    def process_trapezoid(self, cmd):
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid trapezoid command: {cmd}")
        
        points_data = cmd[1]
        if len(points_data) != 4:
            raise RuntimeError(f"Trapezoid requires exactly 4 points, got: {len(points_data)}")

        ps = [Point(p) for p in points_data]
        for p in ps:
            self.register_pt(p)

        instr = Parameter(DiagramType.QUADRILATERAL, ps, QuadrilateralType.TRAPEZOID)
        self.instructions.append(instr)

    def process_rhombus(self, cmd):
        """[NEW] Process DSL: (rhombus (A B C D))"""
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid rhombus command: {cmd}")
        
        points_data = cmd[1]
        if len(points_data) != 4:
            raise RuntimeError(f"Rhombus requires exactly 4 points, got: {len(points_data)}")

        ps = [Point(p) for p in points_data]
        for p in ps:
            self.register_pt(p)

        # Mapping sang kiểu RHOMBUS
        instr = Parameter(DiagramType.QUADRILATERAL, ps, QuadrilateralType.RHOMBUS)
        self.instructions.append(instr)

    def process_define(self, cmd):
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

        processed_args = []
        for arg in construction_args:
            if isinstance(arg, tuple):
                processed_args.extend(arg[1:])
            else:
                processed_args.append(arg)

        instr = Parameter(
            diagram_type=DiagramType.POINT,
            objects=[Point(point_name)],
            param_type=construction_type,
            args=tuple(Point(p) for p in processed_args)
        )
        self.instructions.append(instr)

    def process_circle(self, cmd):
        if len(cmd) < 3:
            raise RuntimeError(f"Invalid circle command: {cmd}")

        center_name = cmd[1]
        construction = cmd[2]

        if not isinstance(construction, tuple):
            raise RuntimeError(f"Circle construction must be a tuple: {construction}")

        construction_type = construction[0].lower()
        construction_args = construction[1:]

        instr = Parameter(
            diagram_type=DiagramType.CIRCLE,
            objects=[Point(center_name)],
            param_type=construction_type,
            args=tuple(Point(p) for p in construction_args)
        )
        self.instructions.append(instr)

    def process_segment(self, cmd):
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