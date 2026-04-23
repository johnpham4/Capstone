<<<<<<< HEAD

from typing import List, Tuple, Any
from loguru import logger

from src.services.diagram.dsl_parser import DSLParser
from .model.value_objects import Point
from .model.instructions import Parameter, Assertion, DistanceValue
from .model.types import DiagramType, TriangleType, QuadrilateralType


class DiagramBuilder:
    def __init__(self, problem_lines: List[str]):
        self.points: List[Point] = []
        self.instructions: List[Any] = []
        self.problem_lines = problem_lines
        self.warnings: List[str] = []  # Track skipped commands

        cmds = DSLParser.parse_sexprs(self.problem_lines)
        for idx, cmd in enumerate(cmds):
            try:
                self.process_command(cmd)
            except Exception as e:
                # Log warning but continue processing
                warning_msg = f"Line {idx+1}: {cmd} - Error: {str(e)}"
                self.warnings.append(warning_msg)
                logger.warning(f"SKIPPED {warning_msg}")

        # Normalize noisy LLM DSL:
        # (triangle (A B C)) + (equal-distance A B A C) => explicit isosceles at A.
        self._promote_plain_triangles_from_equal_distance()

        # Transform diameter + existing circle => diameter-point construction
        # (diameter A K O) + (circle O ...) => (define K point (diameter-point A O))
        self._transform_diameter_with_existing_circle()

    def _extract_shared_triangle_pattern(self, point_names: List[str]):
        """Return (shared_vertex, side_end_1, side_end_2) for AB=AC-like patterns."""
        if len(point_names) != 4:
            return None

        p1, p2, p3, p4 = point_names
        patterns = (
            (p1 == p3 and p2 != p4, p1, p2, p4),
            (p1 == p4 and p2 != p3, p1, p2, p3),
            (p2 == p3 and p1 != p4, p2, p1, p4),
            (p2 == p4 and p1 != p3, p2, p1, p3),
        )

        for matched, shared, end_1, end_2 in patterns:
            if matched and len({shared, end_1, end_2}) == 3:
                return shared, end_1, end_2

        return None

    def _promote_plain_triangles_from_equal_distance(self):
        """Promote plain `(triangle ...)` to explicit isosceles when equal-distance defines two equal sides."""
        plain_triangle_by_vertices = {}

        for instr in self.instructions:
            if (
                isinstance(instr, Parameter)
                and instr.diagram_type == DiagramType.TRIANGLE
                and len(instr.objects) == 3
                and instr.param_type is None
            ):
                key = frozenset(obj.val for obj in instr.objects)
                plain_triangle_by_vertices.setdefault(key, []).append(instr)

        for instr in self.instructions:
            if (
                not isinstance(instr, Assertion)
                or instr.constraint_type != 'equal_distance'
                or len(instr.objects) != 4
            ):
                continue

            point_names = [obj.val for obj in instr.objects if hasattr(obj, 'val')]
            if len(point_names) != 4:
                continue

            pattern = self._extract_shared_triangle_pattern(point_names)
            if pattern is None:
                continue

            shared_vertex, side_end_1, side_end_2 = pattern
            candidate_key = frozenset((shared_vertex, side_end_1, side_end_2))
            candidates = plain_triangle_by_vertices.get(candidate_key, [])

            for tri_instr in candidates:
                if tri_instr.param_type is None:
                    tri_instr.param_type = TriangleType.ISOSCELES
                    tri_instr.args = (Point(shared_vertex),)
                    break

    def _transform_diameter_with_existing_circle(self):
        """Transform diameter assertions when the center already has a defined circle.

        When (diameter P1 P2 Center) is used and (circle Center ...) exists:
        - If an endpoint is NOT yet defined: create a 'diameter-point' Parameter
          (K = 2*Center - KnownEndpoint, i.e. reflection through center)
        - Remove the original diameter assertion (the construction handles everything)

        When center does NOT have an existing circle, keep the diameter assertion as-is
        (it will create a new circle in the optimizer).
        """
        # 1. Collect circle centers from circle Parameters
        circle_centers = set()
        for instr in self.instructions:
            if (
                isinstance(instr, Parameter)
                and instr.diagram_type == DiagramType.CIRCLE
            ):
                center_name = instr.objects[0].val
                circle_centers.add(center_name)

        if not circle_centers:
            return

        # 2. Collect registered point names
        registered_point_names = {p.val for p in self.points}

        # 3. Find diameter assertions to transform
        to_remove = []
        to_add = []

        for instr in self.instructions:
            if not isinstance(instr, Assertion) or instr.constraint_type != 'diameter':
                continue

            p1_name = instr.objects[0].val  # endpoint 1
            p2_name = instr.objects[1].val  # endpoint 2
            center_name = instr.objects[2].val  # center

            # Only transform if center matches an existing circle
            if center_name not in circle_centers:
                continue

            p1_defined = p1_name in registered_point_names
            p2_defined = p2_name in registered_point_names

            if not p1_defined and not p2_defined:
                # Both undefined — can't determine which is the reference point,
                # fall back to the original diameter assertion behavior
                continue

            # Mark original diameter assertion for removal
            to_remove.append(instr)

            if not p2_defined:
                # P2 is undefined — define as diameter-point (reflection of P1 through Center)
                # K = 2*Center - P1
                new_point = Point(p2_name)
                self.register_pt(new_point)
                to_add.append(Parameter(
                    diagram_type=DiagramType.POINT,
                    objects=[new_point],
                    param_type="diameter-point",
                    args=(Point(p1_name), Point(center_name))
                ))
                logger.info(
                    f"DSL transform: (diameter {p1_name} {p2_name} {center_name}) "
                    f"-> (define {p2_name} point (diameter-point {p1_name} {center_name}))"
                )
            elif not p1_defined:
                # P1 is undefined — define as diameter-point (reflection of P2 through Center)
                new_point = Point(p1_name)
                self.register_pt(new_point)
                to_add.append(Parameter(
                    diagram_type=DiagramType.POINT,
                    objects=[new_point],
                    param_type="diameter-point",
                    args=(Point(p2_name), Point(center_name))
                ))
                logger.info(
                    f"DSL transform: (diameter {p1_name} {p2_name} {center_name}) "
                    f"-> (define {p1_name} point (diameter-point {p2_name} {center_name}))"
                )
            else:
                # Both endpoints are already defined.
                # If they are vertices of a triangle with circumcenter = center,
                # promote the triangle to RIGHT at the third vertex (Thales' theorem:
                # if AB is a diameter of the circumcircle, then angle C = 90°).
                promoted = False
                for tri_instr in self.instructions:
                    if (
                        isinstance(tri_instr, Parameter)
                        and tri_instr.diagram_type == DiagramType.TRIANGLE
                        and len(tri_instr.objects) == 3
                    ):
                        vertex_names = [obj.val for obj in tri_instr.objects]
                        if p1_name in vertex_names and p2_name in vertex_names:
                            # Found triangle containing both endpoints
                            # Find the third vertex (the one NOT on the diameter)
                            third_vertex = None
                            for obj in tri_instr.objects:
                                if obj.val != p1_name and obj.val != p2_name:
                                    third_vertex = obj.val
                                    break

                            if third_vertex:
                                # Promote to right triangle at third vertex
                                tri_instr.param_type = TriangleType.RIGHT
                                tri_instr.args = (Point(third_vertex),)
                                promoted = True
                                logger.info(
                                    f"DSL transform: (diameter {p1_name} {p2_name} {center_name}) "
                                    f"-> triangle promoted to RIGHT at {third_vertex}"
                                )
                                break

                if not promoted:
                    # Fallback: add lightweight collinear constraint
                    to_add.append(Assertion(
                        constraint_type='diameter_collinear',
                        objects=[Point(center_name), Point(p1_name), Point(p2_name)]
                    ))
                    logger.info(
                        f"DSL transform: (diameter {p1_name} {p2_name} {center_name}) "
                        f"-> (diameter_collinear {center_name} {p1_name} {p2_name})"
                    )

        # 4. Apply transformations
        for instr in to_remove:
            self.instructions.remove(instr)
        self.instructions.extend(to_add)

    def process_command(self, cmd: Tuple):
        if not isinstance(cmd[0], str):
            raise RuntimeError("Command must start with a string")

        head = cmd[0].lower()

        if head == "triangle":
            self.process_triangle(cmd)
        elif head in ["quadrilateral", "square", "rectangle", "trapezoid", "parallelogram", "rhombus"]:
            self.process_quadrilateral(cmd)
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
        elif head == "tangent":
            self.process_tangent(cmd)
        elif head == "diameter":
            self.process_diameter(cmd)
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


    def process_quadrilateral(self, cmd):
        """Process quadrilateral commands: (quadrilateral (A B C D)), (square (A B C D)), (rectangle (A B C D)), etc."""
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid quadrilateral command: {cmd}")

        # Determine quadrilateral type from command name
        quad_type_str = cmd[0].upper()

        # Generic quadrilateral has no specific type
        if quad_type_str == "QUADRILATERAL":
            quad_type = None
        else:
            try:
                quad_type = QuadrilateralType[quad_type_str]
            except KeyError:
                raise RuntimeError(f"Unknown quadrilateral type: {cmd[0]}")

        ps = [Point(p) for p in cmd[1]]

        if len(ps) != 4:
            raise ValueError(f"Quadrilateral must have 4 vertices, got {len(ps)}")

        # Register all 4 points
        for p in ps:
            self.register_pt(p)

        # If no additional parameters
        if len(cmd) == 2:
            instr = Parameter(DiagramType.QUADRILATERAL, ps, quad_type)
            self.instructions.append(instr)
            return

        # With additional parameters: (square (A B C D) (param_type args...))
        param_method = cmd[2]

        if isinstance(param_method, tuple) and len(param_method) > 0:
            param_type_lower = param_method[0].lower()
            # Process arguments - convert to Points if they are strings
            args = tuple(Point(p) if isinstance(p, str) else p for p in param_method[1:])
            instr = Parameter(DiagramType.QUADRILATERAL, ps, quad_type, args)
        else:
            instr = Parameter(DiagramType.QUADRILATERAL, ps, quad_type)

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
        - (circle O) -> auto infer circle from known center type (fallback radius 1.0)
        - (circle O (radius 0.5)) -> explicit radius
        - (circle O (incircle A B C)) -> inscribed circle
        """
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid circle command: {cmd}")

        center_name = cmd[1]

        # If only (circle O) - use default radius
        if len(cmd) == 2:
            construction_type = 'auto'
            construction_args = ()
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

    def process_tangent(self, cmd):
        """Process: (tangent M (circle O) AB) -> Line AB is tangent to circle O at point M"""
        if len(cmd) != 4:
            raise RuntimeError(f"tangent requires 3 arguments (tangent-point, circle, line-points): {cmd}")

        # Extract tangent point M
        tangent_point = Point(cmd[1])

        # Extract circle center O from nested tuple (circle O)
        circle_spec = cmd[2]
        if not isinstance(circle_spec, tuple) or circle_spec[0] != "circle":
            raise RuntimeError(f"tangent requires (circle O) as second argument: {cmd}")
        circle_center = Point(circle_spec[1])

        # Extract line points from AB (two-character string)
        line_points_str = cmd[3]
        if not isinstance(line_points_str, str) or len(line_points_str) != 2:
            raise RuntimeError(f"tangent requires two-character line points (e.g., 'AB'): {cmd}")
        point_a = Point(line_points_str[0])
        point_b = Point(line_points_str[1])

        # Create assertion: [tangent_point, circle_center, line_point_a, line_point_b]
        instr = Assertion(
            constraint_type='tangent',
            objects=[tangent_point, circle_center, point_a, point_b]
        )
        self.instructions.append(instr)

    def process_diameter(self, cmd):
        """Process: (diameter M N O) -> MN is diameter of circle O"""
        if len(cmd) != 4:
            raise RuntimeError(f"diameter requires 3 points (p1, p2, center): {cmd}")

        p1 = Point(cmd[1])  # M
        p2 = Point(cmd[2])  # N
        center = Point(cmd[3])  # O

        # Diameter constraint: MN is diameter of circle O
        # - M and N both on circle O
        # - O is midpoint of M and N
        # - M-O-N are collinear
        instr = Assertion(
            constraint_type='diameter',
            objects=[p1, p2, center]
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
        """
        Process equal-distance constraint:
        - (equal-distance A B C D) -> Distance AB = Distance CD (4 points)
        - (equal-distance A B 1.0) -> Distance AB = 1.0 (2 points + fixed value)
        """
        if len(cmd) == 4:
            # Format: (equal-distance A B 1.0) - fixed distance
            try:
                p1 = Point(cmd[1])
                p2 = Point(cmd[2])
                distance_value = float(cmd[3])

                instr = Assertion(
                    constraint_type='fixed_distance',
                    objects=[p1, p2],
                    distance=DistanceValue(distance_value)
                )
                self.instructions.append(instr)

            except (ValueError, TypeError):
                raise RuntimeError(f"equal-distance with 3 params requires (point point number): {cmd}")

        elif len(cmd) == 5:
            # Format: (equal-distance A B C D) - equal distance between two segments
            p1 = Point(cmd[1])
            p2 = Point(cmd[2])
            p3 = Point(cmd[3])
            p4 = Point(cmd[4])

            instr = Assertion(
                constraint_type='equal_distance',
                objects=[p1, p2, p3, p4]
            )
            self.instructions.append(instr)

        else:
            raise RuntimeError(f"equal-distance requires either 3 params (p1 p2 distance) or 4 points (p1 p2 p3 p4): {cmd}")

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

=======
from typing import List, Tuple, Any

from src.services.diagram.dsl_parser import DSLParser
from src.models.domain.geometry import Point
from src.models.domain.geometry.instructions import Parameter, DistanceValue, Assertion
from src.models.domain.geometry.types import DiagramType, QuadrilateralType, TriangleType


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
        elif head in ["quadrilateral", "square", "rectangle", "trapezoid", "parallelogram", "rhombus"]:
            self.process_quadrilateral(cmd)
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

    def process_quadrilateral(self, cmd):
        """Process quadrilateral commands: (quadrilateral (A B C D)), (square (A B C D)), (rectangle (A B C D)), etc."""
        if len(cmd) < 2:
            raise RuntimeError(f"Invalid quadrilateral command: {cmd}")

        # Determine quadrilateral type from command name
        quad_type_str = cmd[0].upper()
        
        # Generic quadrilateral has no specific type
        if quad_type_str == "QUADRILATERAL":
            quad_type = None
        else:
            try:
                quad_type = QuadrilateralType[quad_type_str]
            except KeyError:
                raise RuntimeError(f"Unknown quadrilateral type: {cmd[0]}")

        ps = [Point(p) for p in cmd[1]]

        if len(ps) != 4:
            raise ValueError(f"Quadrilateral must have 4 vertices, got {len(ps)}")

        # Register all 4 points
        for p in ps:
            self.register_pt(p)

        # If no additional parameters
        if len(cmd) == 2:
            instr = Parameter(DiagramType.QUADRILATERAL, ps, quad_type)
            self.instructions.append(instr)
            return

        # With additional parameters: (square (A B C D) (param_type args...))
        param_method = cmd[2]

        if isinstance(param_method, tuple) and len(param_method) > 0:
            param_type_lower = param_method[0].lower()
            # Process arguments - convert to Points if they are strings
            args = tuple(Point(p) if isinstance(p, str) else p for p in param_method[1:])
            instr = Parameter(DiagramType.QUADRILATERAL, ps, quad_type, args)
        else:
            instr = Parameter(DiagramType.QUADRILATERAL, ps, quad_type)

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
            pt = Point(point_name)
            self.register_pt(pt)
            instr = Parameter(
                diagram_type=DiagramType.POINT,
                objects=[pt],
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
        pt = Point(point_name)
        self.register_pt(pt)
        instr = Parameter(
            diagram_type=DiagramType.POINT,
            objects=[pt],
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
>>>>>>> Dka
