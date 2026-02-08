from typing import List, Tuple, Any

from src.services.diagram.dsl_parser import DSLParser
from src.models.domain.geometry.value_objects import Point
from src.models.domain.geometry.instructions import Parameter, Assertion, DistanceValue
from src.models.domain.geometry.types import DiagramType, TriangleType, QuadrilateralType


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
        elif head == "on-circle":
            self.process_on_circle(cmd)
        elif head == "on-segment":
            self.process_on_segment(cmd)
        elif head == "distance":
            self.process_distance(cmd)
        elif head == "equal-distance":
            self.process_equal_distance(cmd)
        elif head == "parallel":
            self.process_parallel(cmd)
        elif head == "perpendicular":
            self.process_perpendicular(cmd)
        elif head == "angle-equal":
            self.process_angle_equal(cmd)
        elif head == "angle-measure":
            self.process_angle_measure(cmd)
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
        if len(cmd) < 3:
            raise RuntimeError(f"Invalid define command: {cmd}")

        point_name = cmd[1]
        obj_type = cmd[2]

        if obj_type != "point":
            raise NotImplementedError(f"Only 'point' type supported, got: {obj_type}")

        # Simple point definition: (define O point)
        if len(cmd) == 3:
            p = Point(point_name)
            self.register_pt(p)
            # Create simple coords parameter
            instr = Parameter(
                diagram_type=DiagramType.POINT,
                objects=[p],
                param_type="coords",
                args=()
            )
            self.instructions.append(instr)
            return

        # Point with construction: (define H point (projection C (segment A B)))
        construction = cmd[3]
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
        """
        Process circle commands:
        - (circle O) -> default radius 1.0
        - (circle O (radius 0.5)) -> explicit radius
        - (circle O (incircle A B C)) -> inscribed circle
        """
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid circle command: {cmd}")

        center_name = cmd[1]
        
        # If only (circle O) - use default radius
        if len(cmd) == 2:
            construction_type = 'radius'
            construction_args = (1.0,)
        else:
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
            args=tuple(Point(p) if isinstance(p, str) else p for p in construction_args)
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

    def process_on_circle(self, cmd):
        """Process: (on-circle B O) -> Point B lies on circle centered at O"""
        if len(cmd) != 3:
            raise RuntimeError(f"on-circle requires 2 points (point, center): {cmd}")
        
        point = Point(cmd[1])  # B
        center = Point(cmd[2])  # O
        
        instr = Assertion(
            constraint_type='on_circle',
            objects=[point, center]
        )
        self.instructions.append(instr)

    def process_on_segment(self, cmd):
        """Process: (on-segment M C D) -> M lies on segment CD"""
        if len(cmd) != 4:
            raise RuntimeError(f"on-segment requires 3 points (point, seg_p1, seg_p2): {cmd}")
        
        point = Point(cmd[1])  # M
        seg_p1 = Point(cmd[2])  # C
        seg_p2 = Point(cmd[3])  # D
        
        instr = Assertion(
            constraint_type='on_segment',
            objects=[point, seg_p1, seg_p2]
        )
        self.instructions.append(instr)

    def process_distance(self, cmd):
        """Process: (distance O A 0.03) -> Distance OA = 0.03"""
        if len(cmd) != 4:
            raise RuntimeError(f"distance requires 2 points and 1 value: {cmd}")
        
        p1 = Point(cmd[1])  # O
        p2 = Point(cmd[2])  # A
        distance_value = cmd[3]  # 0.03
        
        instr = Assertion(
            constraint_type='distance',
            objects=[p1, p2, DistanceValue(distance_value)]
        )
        self.instructions.append(instr)

    def process_equal_distance(self, cmd):
        """Process: (equal-distance O M O H) -> Distance OM = Distance OH"""
        if len(cmd) != 5:
            raise RuntimeError(f"equal-distance requires 4 points (p1 p2 p3 p4): {cmd}")
        
        p1 = Point(cmd[1])
        p2 = Point(cmd[2])
        p3 = Point(cmd[3])
        p4 = Point(cmd[4])
        
        instr = Assertion(
            constraint_type='equal_distance',
            objects=[p1, p2, p3, p4]
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

        p1 = Point(seg1[1])
        p2 = Point(seg1[2])
        p3 = Point(seg2[1])
        p4 = Point(seg2[2])

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

        p1 = Point(seg1[1])
        p2 = Point(seg1[2])
        p3 = Point(seg2[1])
        p4 = Point(seg2[2])

        instr = Assertion(
            constraint_type='perpendicular',
            objects=[p1, p2, p3, p4]
        )
        self.instructions.append(instr)

    def process_angle_equal(self, cmd):
        """Process: (angle-equal A B C D E F) -> ∠ABC = ∠DEF"""
        if len(cmd) != 7:
            raise RuntimeError(f"angle-equal requires 6 points: {cmd}")

        points = [Point(cmd[i]) for i in range(1, 7)]

        instr = Assertion(
            constraint_type='angle_equal',
            objects=points
        )
        self.instructions.append(instr)

    def process_angle_measure(self, cmd):
        """Process: (angle-measure A B C 120) -> ∠ABC = 120°"""
        if len(cmd) != 5:
            raise RuntimeError(f"angle-measure requires 3 points + 1 degree value: {cmd}")

        points = [Point(cmd[i]) for i in range(1, 4)]
        degrees = cmd[4]
        
        class DegreeValue:
            def __init__(self, value):
                self.val = str(value)
        
        instr = Assertion(
            constraint_type='angle_measure',
            objects=points + [DegreeValue(degrees)]
        )
        self.instructions.append(instr)