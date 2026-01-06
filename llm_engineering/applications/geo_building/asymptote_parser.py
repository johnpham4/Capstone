"""Parser for Asymptote-style geometry DSL"""
from typing import List, Tuple, Union
from instruction import Assert, Compute, Sample, Parameterize
from primitives import Point, Line
from constraint import Constraint
from util import FuncInfo


class AsymptoteParser:
    """Parse Asymptote-style geometry commands"""

    def __init__(self, lines: List[str]):
        self.lines = lines
        self.points = []
        self.lines_list = []
        self.instructions = []
        self.segments = []
        self.seg_colors = []

        self.parse_all()

    def parse_all(self):
        """Parse all lines"""
        for line in self.lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            tokens = self.tokenize(line)
            if tokens:
                self.parse_command(tokens)

    def tokenize(self, s: str) -> List[Union[str, List]]:
        """Convert string to nested tokens"""
        result = []
        current = []
        stack = [result]

        s = s.replace('(', ' ( ').replace(')', ' ) ')
        tokens = s.split()

        for token in tokens:
            if token == '(':
                new_list = []
                stack[-1].append(new_list)
                stack.append(new_list)
            elif token == ')':
                if len(stack) > 1:
                    stack.pop()
            else:
                stack[-1].append(token)

        return result[0] if result else []

    def parse_command(self, tokens):
        """Parse a single command"""
        if not tokens or not isinstance(tokens, list):
            return

        head = tokens[0].lower()

        if head == 'triangle':
            self.parse_triangle(tokens)
        elif head == 'point':
            self.parse_point(tokens)
        elif head == 'free-point':
            self.parse_free_point(tokens)
        elif head == 'line':
            self.parse_line(tokens)
        elif head == 'assert':
            self.parse_assert(tokens)
        else:
            raise ValueError(f"Unknown command: {head}")

    def parse_triangle(self, tokens):
        """Parse (triangle A B C) or (triangle A B C (right-at B))"""
        # tokens = ['triangle', 'A', 'B', 'C'] or ['triangle', 'A', 'B', 'C', [...]]
        point_names = []
        special_type = None
        special_point = None

        i = 1
        while i < len(tokens):
            if isinstance(tokens[i], list):
                # Special triangle constraint
                constraint = tokens[i]
                special_type = constraint[0].lower()
                if len(constraint) > 1:
                    special_point = constraint[1]
                break
            else:
                point_names.append(tokens[i])
            i += 1

        # Register points
        ps = [Point(name) for name in point_names]
        for p in ps:
            if p not in self.points:
                self.points.append(p)

        # Create sample instruction
        if special_type:
            if special_type == 'right-at':
                sampler = 'right-tri'
                args = (Point(special_point),)
            elif special_type == 'isosceles':
                sampler = 'iso-tri'
                args = (Point(special_point),)
            else:
                sampler = special_type
                args = ()
        else:
            sampler = 'triangle'
            args = ()

        instr = Sample(ps, sampler, args)
        self.instructions.append(instr)

        # Add triangle segments
        import numpy as np
        color = np.random.rand(3)
        for i in range(len(ps)):
            self.segments.append((ps[i], ps[(i+1) % len(ps)]))
            self.seg_colors.append(color)

    def parse_point(self, tokens):
        """Parse (point P (orthocenter A B C))"""
        # tokens = ['point', 'P', [...]]
        point_name = tokens[1]
        computation = tokens[2]

        p = Point(point_name)
        if p not in self.points:
            self.points.append(p)

        # Parse computation
        comp_func = self.parse_computation(computation)

        instr = Compute(p, comp_func)
        self.instructions.append(instr)

    def parse_free_point(self, tokens):
        """Parse (free-point D (on-segment A B))"""
        # tokens = ['free-point', 'D', [...]]
        point_name = tokens[1]
        constraint = tokens[2]

        p = Point(point_name)
        if p not in self.points:
            self.points.append(p)

        # Parse parameterization constraint
        param = self.parse_parameterization(constraint)

        instr = Parameterize(p, param)
        self.instructions.append(instr)

    def parse_line(self, tokens):
        """Parse (line l_BC (connecting B C))"""
        # tokens = ['line', 'l_BC', [...]]
        line_name = tokens[1]
        definition = tokens[2]

        l = Line(line_name)
        if l not in self.lines_list:
            self.lines_list.append(l)

        # Parse line definition
        def_type = definition[0].lower()

        if def_type == 'connecting':
            # (connecting B C)
            p1 = Point(definition[1])
            p2 = Point(definition[2])
            comp = FuncInfo('line', [p1, p2])

            # THÊM: Vẽ đoạn thẳng thay vì đường thẳng vô hạn
            import numpy as np
            color = np.random.rand(3)
            self.segments.append((p1, p2))
            self.seg_colors.append(color)
        elif def_type == 'perpendicular-through':
            # (perpendicular-through C (segment A B))
            through_point = Point(definition[1])
            segment = definition[2]
            # segment = ['segment', 'A', 'B']
            seg_p1 = Point(segment[1])
            seg_p2 = Point(segment[2])
            seg_line = Line(f"seg_{seg_p1.val}_{seg_p2.val}")
            # Create implicit line for segment
            seg_comp = FuncInfo('line', [seg_p1, seg_p2])
            seg_instr = Compute(seg_line, seg_comp)
            self.instructions.append(seg_instr)
            # Perpendicular line
            comp = FuncInfo('perp', [through_point, seg_line])
        else:
            raise ValueError(f"Unknown line definition: {def_type}")

        instr = Compute(l, comp)
        self.instructions.append(instr)

    def parse_assert(self, tokens):
        """Parse (assert (parallel l_BC l_DE))"""
        # tokens = ['assert', [...]]
        constraint_expr = tokens[1]

        cons = self.parse_constraint(constraint_expr)
        instr = Assert(cons)
        self.instructions.append(instr)

    def parse_computation(self, expr):
        """Parse computation expression like (orthocenter A B C)"""
        if isinstance(expr, list):
            func_name = expr[0].lower()
            args = [Point(arg) if isinstance(arg, str) else arg for arg in expr[1:]]
            return FuncInfo(func_name, args)
        return expr

    def parse_parameterization(self, expr):
        """Parse parameterization like (on-segment A B)"""
        param_type = expr[0].lower()

        if param_type == 'on-segment':
            p1 = Point(expr[1])
            p2 = Point(expr[2])
            return ('on-seg', p1, p2)
        elif param_type == 'on-line':
            line = Line(expr[1])
            return ('on-line', line)
        else:
            raise ValueError(f"Unknown parameterization: {param_type}")

    def parse_constraint(self, expr):
        """Parse constraint expression"""
        cons_type = expr[0].lower()

        if cons_type == 'parallel':
            # (parallel l_BC l_DE)
            l1 = Line(expr[1])
            l2 = Line(expr[2])
            return Constraint('para', [l1, l2], False)

        elif cons_type == 'perpendicular':
            # (perpendicular l_C (segment A B))
            l1 = Line(expr[1])
            if isinstance(expr[2], list) and expr[2][0] == 'segment':
                # Create implicit line from segment
                p1 = Point(expr[2][1])
                p2 = Point(expr[2][2])
                l2 = Line(f"seg_{p1.val}_{p2.val}")
                # Need to compute this line
                comp = FuncInfo('line', [p1, p2])
                instr = Compute(l2, comp)
                self.instructions.append(instr)
            else:
                l2 = Line(expr[2])
            return Constraint('perp', [l1, l2], False)

        elif cons_type == 'right-angle':
            # (right-angle A D E)
            p1 = Point(expr[1])
            p2 = Point(expr[2])
            p3 = Point(expr[3])
            return Constraint('right', [p1, p2, p3], False)

        elif cons_type == 'equal':
            # (equal (angle B A F) (angle C A F))
            angle1 = self.parse_angle(expr[1])
            angle2 = self.parse_angle(expr[2])
            return Constraint('eqangle', [angle1[0], angle1[1], angle1[2],
                                         angle2[0], angle2[1], angle2[2]], False)

        else:
            raise ValueError(f"Unknown constraint: {cons_type}")

    def parse_angle(self, expr):
        """Parse angle expression like (angle B A F)"""
        if expr[0].lower() == 'angle':
            return (Point(expr[1]), Point(expr[2]), Point(expr[3]))
        raise ValueError(f"Invalid angle expression: {expr}")
