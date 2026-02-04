from typing import List, Tuple, Any

from llm_engineering.applications.diagram.dsl_parser import DSLParser
from llm_engineering.domains.geometry import Point
from llm_engineering.domains.geometry.instructions import Parameter, Assertion, DistanceValue
from llm_engineering.domains.geometry.types import DiagramType, QuadrilateralType, TriangleType

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
        elif head == "angle-measure":
            self.process_angle_measure(cmd)
        elif head == "on-segment":
            self.process_on_segment(cmd)
        elif head == "distance":
            self.process_distance(cmd)
        elif head == "equal-distance":
            self.process_equal_distance(cmd)
        elif head == "on-circle":
            self.process_on_circle(cmd)
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
        
        for p in ps:
            self.register_pt(p)
        
        instr = Parameter(
            DiagramType.QUADRILATERAL,
            ps,
            QuadrilateralType.SQUARE,
            (ps[0],)  
        )
        self.instructions.append(instr)

    def process_define(self, cmd):
        """Process: (define G point (centroid A B C)) or (define O point)"""
        if len(cmd) < 3:
            raise RuntimeError(f"Invalid define command: {cmd}")

        point_name = cmd[1]
        obj_type = cmd[2]

        if obj_type != "point":
            raise NotImplementedError(f"Only 'point' type supported, got: {obj_type}")

        # Handle free point with no construction: (define O point)
        if len(cmd) == 3:
            instr = Parameter(
                diagram_type=DiagramType.POINT,
                objects=[Point(point_name)],
                param_type="coords",  
                args=()
            )
            self.instructions.append(instr)
            return

        # (define G point (centroid A B C))
        construction = cmd[3]

        if not isinstance(construction, tuple):
            raise RuntimeError(f"Construction must be a tuple: {construction}")

        construction_type = construction[0].lower()
        construction_args = construction[1:]

        # (projection A (segment B C))
        processed_args = []
        for arg in construction_args:
            if isinstance(arg, tuple):
                # (segment B C) -> [B, C]
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
            construction = ('radius', 1.0)
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

        # Extract 6 points for two angles
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

        # DSL format: (angle-measure A C B 110)
        # Order: p1=A, vertex=C, p2=B
        points = [Point(cmd[i]) for i in range(1, 4)]
        degrees = cmd[4]  
        
        # Create a Point-like object for the degree value to fit into objects list
        class DegreeValue:
            def __init__(self, value):
                self.val = str(value)
        
        instr = Assertion(
            constraint_type='angle_measure',
            objects=points + [DegreeValue(degrees)]
        )
        self.instructions.append(instr)

    def process_on_segment(self, cmd):
        """Process: (on-segment M C D) -> M lies on segment CD"""
        if len(cmd) != 4:
            raise RuntimeError(f"on-segment requires 3 points (point, seg_p1, seg_p2): {cmd}")
        
        # DSL format: (on-segment M C D) means M lies on segment CD
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
        distance_value = cmd[3]  # 0.03 (float or string)
        
        # Convert to DistanceValue wrapper
        instr = Assertion(
            constraint_type='distance',
            objects=[p1, p2, DistanceValue(distance_value)]
        )
        self.instructions.append(instr)

    def process_equal_distance(self, cmd):
        """Process: (equal-distance O M O H) -> Distance OM = Distance OH"""
        if len(cmd) != 5:
            raise RuntimeError(f"equal-distance requires 4 points (p1 p2 p3 p4): {cmd}")
        
        p1 = Point(cmd[1])  # O
        p2 = Point(cmd[2])  # M
        p3 = Point(cmd[3])  # O
        p4 = Point(cmd[4])  # H
        
        instr = Assertion(
            constraint_type='equal_distance',
            objects=[p1, p2, p3, p4]
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
