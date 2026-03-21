<<<<<<< HEAD
import math
from typing import List, Tuple

from networkx import center


class Initializer:

    @staticmethod
    def init_isoceles_triangle(apex_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        # Base configuration: apex at top, base at bottom
        base_coords = [
            (0.0, 0.8 * scale),      # Apex
            (-0.6 * scale, -0.4 * scale),  # Base left
            (0.6 * scale, -0.4 * scale)    # Base right
        ]

        # Rotate to put apex at correct position
        if apex_idx == 0:
            return base_coords
        elif apex_idx == 1:
            return [base_coords[1], base_coords[0], base_coords[2]]
        else:  # apex_idx == 2
            return [base_coords[1], base_coords[2], base_coords[0]]

    @staticmethod
    def init_right_triangle(right_angle_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        # Axis-aligned template (like a corner of a rectangle) to keep right triangles readable.
        leg_x = 1.2 * scale
        leg_y = 0.95 * scale

        # Right angle at first point.
        base_coords = [
            (-leg_x / 2, -leg_y / 2),  # Right angle
            ( leg_x / 2, -leg_y / 2),  # Horizontal leg endpoint
            (-leg_x / 2,  leg_y / 2)   # Vertical leg endpoint
        ]

        if right_angle_idx == 0:
            return base_coords
        elif right_angle_idx == 1:
            return [base_coords[1], base_coords[0], base_coords[2]]
        else:
            return [base_coords[1], base_coords[2], base_coords[0]]

    @staticmethod
    def init_equilateral_triangle(scale: float = 1.35) -> List[Tuple[float, float]]:
        height = math.sqrt(3) / 2 * scale
        return [
            (0.0, 2 * height / 3),           # Top
            (-0.5 * scale, -height / 3),     # Bottom left
            (0.5 * scale, -height / 3)       # Bottom right
        ]

    @staticmethod

    def init_scalene_triangle(scale: float = 1.35) -> List[Tuple[float, float]]:
        """Scalene triangle with all sides different"""
        return [
            (-0.7 * scale, -0.4 * scale),   # A
            (0.8 * scale, -0.2 * scale),    # B
            (-0.1 * scale, 0.9 * scale)     # C: right (không đối xứng)
        ]

    @staticmethod
    def init_right_isoceles_triangle(right_angle_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        # Centered square-corner layout: equal legs and clear diagonal hypotenuse.
        leg_length = 1.1 * scale
        base_coords = [
            (-leg_length / 2, -leg_length / 2),  # Right angle
            ( leg_length / 2, -leg_length / 2),  # Horizontal leg
            (-leg_length / 2,  leg_length / 2)   # Vertical leg
        ]

        if right_angle_idx == 0:
            return base_coords
        elif right_angle_idx == 1:
            return [base_coords[1], base_coords[0], base_coords[2]]
        else:
            return [base_coords[1], base_coords[2], base_coords[0]]


    @staticmethod
    def init_quadrilateral(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Generic quadrilateral with no special properties"""
        return [
            (-0.6 * scale, -0.3 * scale),    # Bottom left
            (0.5 * scale, -0.2 * scale),     # Bottom right
            (0.4 * scale, 0.7 * scale),      # Top right
            (-0.4 * scale, 0.6 * scale)      # Top left
        ]

    @staticmethod
    def init_rectangle(width: float = 1.0, height: float = 0.7) -> List[Tuple[float, float]]:
        """Initialize coordinates for a rectangle"""
        return [
            (-width/2, -height/2),  # Bottom left
            (width/2, -height/2),   # Bottom right
            (width/2, height/2),    # Top right
            (-width/2, height/2)    # Top left
        ]

    @staticmethod
    def init_square(side: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize coordinates for a square"""
        return Initializer.init_rectangle(side, side)

    @staticmethod
    def init_trapezoid(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize coordinates for a trapezoid (one pair of parallel sides)"""
        return [
            (-0.7 * scale, -0.4 * scale),    # Bottom left
            (0.7 * scale, -0.4 * scale),     # Bottom right (longer base)
            (0.4 * scale, 0.4 * scale),      # Top right
            (-0.4 * scale, 0.4 * scale)      # Top left (shorter top)
        ]

    @staticmethod
    def init_parallelogram(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize a visually stable parallelogram for cleaner renders."""
        return [
            (-0.75 * scale, -0.45 * scale),  # Bottom left
            (0.75 * scale, -0.25 * scale),   # Bottom right
            (1.05 * scale, 0.55 * scale),    # Top right
            (-0.45 * scale, 0.35 * scale)    # Top left
        ]

    @staticmethod
    def init_rhombus(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize rhombus (diamond shape with all sides equal)"""
        return [
            (0.0, -0.8 * scale),  # Bottom
            (0.5 * scale, 0.0),   # Right
            (0.0, 0.8 * scale),   # Top
            (-0.5 * scale, 0.0)   # Left
        ]
        
    @staticmethod
    def init_circle_with_positioned_points(center: Tuple[float, float] = (0.0, 0.0),
                                            radius: float = 0.4,
                                            points_distances: List[Tuple[float, float]] = None
                                        ) -> List[Tuple[float, float]]:
        if points_distances is None:
            points_distances = []
        
        result = [center]
        for distance, angle_deg in points_distances:
            angle_rad = math.radians(angle_deg)
            point = (
                center[0] + distance * math.cos(angle_rad),
                center[1] + distance * math.sin(angle_rad)
                )
            result.append(point)
        
        return result
        
    

    @staticmethod
    def init_triangle_incircle(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize equilateral triangle with incenter at origin"""
        h = math.sqrt(3) / 3 * scale
        return [
            (0.0, 2 * h),           # Top
            (-0.5 * scale, -h),     # Bottom left
            (0.5 * scale, -h),      # Bottom right
            (0.0, 0.0)              # Incenter at origin
        ]

    @staticmethod
    def init_triangle_circumcircle(radius: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize triangle inscribed in circle with circumcenter at origin"""
        # 3 points evenly spaced on circle (120 degrees apart)
        angles = [math.pi/2, 7*math.pi/6, 11*math.pi/6]  # 90°, 210°, 330°
        return [
            (radius * math.cos(a), radius * math.sin(a))
            for a in angles
        ] + [(0.0, 0.0)]  # Circumcenter at origin

    @staticmethod
    def init_triangle_with_centroid(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize triangle with centroid at origin"""
        # Any triangle, centroid will be at average of vertices
        return [
            (0.0, 0.6 * scale),      # Top
            (-0.5 * scale, -0.3 * scale),  # Bottom left
            (0.5 * scale, -0.3 * scale),   # Bottom right
            (0.0, 0.0)               # Centroid at origin
        ]

    @staticmethod
    def init_right_triangle_with_orthocenter(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize right triangle with orthocenter at right angle vertex"""
        # For right triangle, orthocenter = right angle vertex
        return [
            (0.0, 0.0),              # Orthocenter (right angle)
            (0.8 * scale, 0.0),      # Along x-axis
            (0.0, 0.6 * scale),      # Along y-axis
            (0.0, 0.0)               # Orthocenter position
        ]

    @staticmethod
    def init_triangle_with_angle_bisector(apex_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        """
        Initialize triangle with angle bisector from apex
        """
        base_coords = [
            (0.0, 0.8 * scale),           # Apex A
            (-0.6 * scale, -0.4 * scale), # Base left B
            (0.6 * scale, -0.4 * scale),  # Base right C
        ]

        # D is midpoint of BC (for isosceles, bisector = median)
        d_x = (base_coords[1][0] + base_coords[2][0]) / 2
        d_y = (base_coords[1][1] + base_coords[2][1]) / 2

        bisector_coords = base_coords + [(d_x, d_y)]

        # Rotate based on apex_idx
        if apex_idx == 0:
            return bisector_coords
        elif apex_idx == 1:
            return [bisector_coords[1], bisector_coords[0], bisector_coords[2], bisector_coords[3]]
        else:
            return [bisector_coords[2], bisector_coords[0], bisector_coords[1], bisector_coords[3]]
        
    @staticmethod
    def init_line_circle_intersection(center: Tuple[float, float],
                                      radius: float,
                                      line_point1: Tuple[float, float],
                                      line_point2: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Tính 2 giao điểm của đường thẳng qua line_point1, line_point2 với đường tròn (center, radius).
        Trả về [intersection1, intersection2] hoặc [] nếu không có giao điểm.
        """
        cx, cy = center
        x1, y1 = line_point1
        x2, y2 = line_point2
        
        # Direction vector của đường thẳng
        dx = x2 - x1
        dy = y2 - y1
        
        # Tránh division by zero
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return []  # Line points trùng nhau
        
        # Vector từ line_point1 đến center
        fx = x1 - cx
        fy = y1 - cy
        
        # Giải phương trình bậc 2: a*t^2 + b*t + c = 0
        # Điểm trên line: (x1 + t*dx, y1 + t*dy)
        # Khoảng cách đến center = radius
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = fx*fx + fy*fy - radius*radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            # Không có giao điểm, trả về 2 điểm gần nhất trên line
            # Điểm gần nhất là khi đạo hàm = 0
            t_closest = -b / (2*a)
            closest_x = x1 + t_closest * dx
            closest_y = y1 + t_closest * dy
            # Trả về 2 điểm cách closest một chút
            offset = 0.1
            return [
                (closest_x - offset*dx, closest_y - offset*dy),
                (closest_x + offset*dx, closest_y + offset*dy)
            ]
        
        # Có giao điểm
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2*a)
        t2 = (-b + sqrt_d) / (2*a)
        
        p1 = (x1 + t1*dx, y1 + t1*dy)
        p2 = (x1 + t2*dx, y1 + t2*dy)
        
        return [p1, p2]

    @staticmethod
    def init_obtuse_triangle(apex_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        """Initialize an obtuse triangle"""
        coords = [
            (0.0 * scale, 0.0 * scale),
            (1.0 * scale, 0.0 * scale),
            (-0.3 * scale, 0.5 * scale),
        ]
        if apex_idx != 0:
            coords = [coords[apex_idx]] + [coords[i] for i in range(3) if i != apex_idx]
        return coords

    @staticmethod
    def add_noise(coords: List[Tuple[float, float]], noise_scale: float = 0.05) -> List[Tuple[float, float]]:
        import random
        return [
            (x + random.uniform(-noise_scale, noise_scale),
             y + random.uniform(-noise_scale, noise_scale))
            for x, y in coords
        ]
=======
import math
from typing import List, Tuple

from networkx import center


class Initializer:

    @staticmethod
    def init_isoceles_triangle(apex_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        # Base configuration: apex at top, base at bottom
        base_coords = [
            (0.0, 0.8 * scale),      # Apex
            (-0.6 * scale, -0.4 * scale),  # Base left
            (0.6 * scale, -0.4 * scale)    # Base right
        ]

        # Rotate to put apex at correct position
        if apex_idx == 0:
            return base_coords
        elif apex_idx == 1:
            return [base_coords[1], base_coords[0], base_coords[2]]
        else:  # apex_idx == 2
            return [base_coords[1], base_coords[2], base_coords[0]]

    @staticmethod
    def init_right_triangle(right_angle_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        # Axis-aligned template (like a corner of a rectangle) to keep right triangles readable.
        leg_x = 1.2 * scale
        leg_y = 0.95 * scale

        # Right angle at first point.
        base_coords = [
            (-leg_x / 2, -leg_y / 2),  # Right angle
            ( leg_x / 2, -leg_y / 2),  # Horizontal leg endpoint
            (-leg_x / 2,  leg_y / 2)   # Vertical leg endpoint
        ]

        if right_angle_idx == 0:
            return base_coords
        elif right_angle_idx == 1:
            return [base_coords[1], base_coords[0], base_coords[2]]
        else:
            return [base_coords[1], base_coords[2], base_coords[0]]

    @staticmethod
    def init_equilateral_triangle(scale: float = 1.35) -> List[Tuple[float, float]]:
        height = math.sqrt(3) / 2 * scale
        return [
            (0.0, 2 * height / 3),           # Top
            (-0.5 * scale, -height / 3),     # Bottom left
            (0.5 * scale, -height / 3)       # Bottom right
        ]

    @staticmethod

    def init_scalene_triangle(scale: float = 1.35) -> List[Tuple[float, float]]:
        """Scalene triangle with all sides different"""
        return [
            (-0.7 * scale, -0.4 * scale),   # A
            (0.8 * scale, -0.2 * scale),    # B
            (-0.1 * scale, 0.9 * scale)     # C: right (không đối xứng)
        ]

    @staticmethod
    def init_right_isoceles_triangle(right_angle_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        # Centered square-corner layout: equal legs and clear diagonal hypotenuse.
        leg_length = 1.1 * scale
        base_coords = [
            (-leg_length / 2, -leg_length / 2),  # Right angle
            ( leg_length / 2, -leg_length / 2),  # Horizontal leg
            (-leg_length / 2,  leg_length / 2)   # Vertical leg
        ]

        if right_angle_idx == 0:
            return base_coords
        elif right_angle_idx == 1:
            return [base_coords[1], base_coords[0], base_coords[2]]
        else:
            return [base_coords[1], base_coords[2], base_coords[0]]


    @staticmethod
    def init_quadrilateral(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Generic quadrilateral with no special properties"""
        return [
            (-0.6 * scale, -0.3 * scale),    # Bottom left
            (0.5 * scale, -0.2 * scale),     # Bottom right
            (0.4 * scale, 0.7 * scale),      # Top right
            (-0.4 * scale, 0.6 * scale)      # Top left
        ]

    @staticmethod
    def init_rectangle(width: float = 1.0, height: float = 0.7) -> List[Tuple[float, float]]:
        """Initialize coordinates for a rectangle"""
        return [
            (-width/2, -height/2),  # Bottom left
            (width/2, -height/2),   # Bottom right
            (width/2, height/2),    # Top right
            (-width/2, height/2)    # Top left
        ]

    @staticmethod
    def init_square(side: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize coordinates for a square"""
        return Initializer.init_rectangle(side, side)

    @staticmethod
    def init_trapezoid(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize coordinates for a trapezoid (one pair of parallel sides)"""
        return [
            (-0.7 * scale, -0.4 * scale),    # Bottom left
            (0.7 * scale, -0.4 * scale),     # Bottom right (longer base)
            (0.4 * scale, 0.4 * scale),      # Top right
            (-0.4 * scale, 0.4 * scale)      # Top left (shorter top)
        ]

    @staticmethod
    def init_parallelogram(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize a visually stable parallelogram for cleaner renders."""
        return [
            (-0.75 * scale, -0.45 * scale),  # Bottom left
            (0.75 * scale, -0.25 * scale),   # Bottom right
            (1.05 * scale, 0.55 * scale),    # Top right
            (-0.45 * scale, 0.35 * scale)    # Top left
        ]

    @staticmethod
    def init_rhombus(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize rhombus (diamond shape with all sides equal)"""
        return [
            (0.0, -0.8 * scale),  # Bottom
            (0.5 * scale, 0.0),   # Right
            (0.0, 0.8 * scale),   # Top
            (-0.5 * scale, 0.0)   # Left
        ]
        
    @staticmethod
    def init_circle_with_positioned_points(center: Tuple[float, float] = (0.0, 0.0),
                                            radius: float = 0.4,
                                            points_distances: List[Tuple[float, float]] = None
                                        ) -> List[Tuple[float, float]]:
        if points_distances is None:
            points_distances = []
        
        result = [center]
        for distance, angle_deg in points_distances:
            angle_rad = math.radians(angle_deg)
            point = (
                center[0] + distance * math.cos(angle_rad),
                center[1] + distance * math.sin(angle_rad)
                )
            result.append(point)
        
        return result
        
    

    @staticmethod
    def init_triangle_incircle(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize equilateral triangle with incenter at origin"""
        h = math.sqrt(3) / 3 * scale
        return [
            (0.0, 2 * h),           # Top
            (-0.5 * scale, -h),     # Bottom left
            (0.5 * scale, -h),      # Bottom right
            (0.0, 0.0)              # Incenter at origin
        ]

    @staticmethod
    def init_triangle_circumcircle(radius: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize triangle inscribed in circle with circumcenter at origin"""
        # 3 points evenly spaced on circle (120 degrees apart)
        angles = [math.pi/2, 7*math.pi/6, 11*math.pi/6]  # 90°, 210°, 330°
        return [
            (radius * math.cos(a), radius * math.sin(a))
            for a in angles
        ] + [(0.0, 0.0)]  # Circumcenter at origin

    @staticmethod
    def init_triangle_with_centroid(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize triangle with centroid at origin"""
        # Any triangle, centroid will be at average of vertices
        return [
            (0.0, 0.6 * scale),      # Top
            (-0.5 * scale, -0.3 * scale),  # Bottom left
            (0.5 * scale, -0.3 * scale),   # Bottom right
            (0.0, 0.0)               # Centroid at origin
        ]

    @staticmethod
    def init_right_triangle_with_orthocenter(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize right triangle with orthocenter at right angle vertex"""
        # For right triangle, orthocenter = right angle vertex
        return [
            (0.0, 0.0),              # Orthocenter (right angle)
            (0.8 * scale, 0.0),      # Along x-axis
            (0.0, 0.6 * scale),      # Along y-axis
            (0.0, 0.0)               # Orthocenter position
        ]

    @staticmethod
    def init_triangle_with_angle_bisector(apex_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        """
        Initialize triangle with angle bisector from apex
        """
        base_coords = [
            (0.0, 0.8 * scale),           # Apex A
            (-0.6 * scale, -0.4 * scale), # Base left B
            (0.6 * scale, -0.4 * scale),  # Base right C
        ]

        # D is midpoint of BC (for isosceles, bisector = median)
        d_x = (base_coords[1][0] + base_coords[2][0]) / 2
        d_y = (base_coords[1][1] + base_coords[2][1]) / 2

        bisector_coords = base_coords + [(d_x, d_y)]

        # Rotate based on apex_idx
        if apex_idx == 0:
            return bisector_coords
        elif apex_idx == 1:
            return [bisector_coords[1], bisector_coords[0], bisector_coords[2], bisector_coords[3]]
        else:
            return [bisector_coords[2], bisector_coords[0], bisector_coords[1], bisector_coords[3]]
        
    @staticmethod
    def init_line_circle_intersection(center: Tuple[float, float],
                                      radius: float,
                                      line_point1: Tuple[float, float],
                                      line_point2: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Tính 2 giao điểm của đường thẳng qua line_point1, line_point2 với đường tròn (center, radius).
        Trả về [intersection1, intersection2] hoặc [] nếu không có giao điểm.
        """
        cx, cy = center
        x1, y1 = line_point1
        x2, y2 = line_point2
        
        # Direction vector của đường thẳng
        dx = x2 - x1
        dy = y2 - y1
        
        # Tránh division by zero
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return []  # Line points trùng nhau
        
        # Vector từ line_point1 đến center
        fx = x1 - cx
        fy = y1 - cy
        
        # Giải phương trình bậc 2: a*t^2 + b*t + c = 0
        # Điểm trên line: (x1 + t*dx, y1 + t*dy)
        # Khoảng cách đến center = radius
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = fx*fx + fy*fy - radius*radius
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            # Không có giao điểm, trả về 2 điểm gần nhất trên line
            # Điểm gần nhất là khi đạo hàm = 0
            t_closest = -b / (2*a)
            closest_x = x1 + t_closest * dx
            closest_y = y1 + t_closest * dy
            # Trả về 2 điểm cách closest một chút
            offset = 0.1
            return [
                (closest_x - offset*dx, closest_y - offset*dy),
                (closest_x + offset*dx, closest_y + offset*dy)
            ]
        
        # Có giao điểm
        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2*a)
        t2 = (-b + sqrt_d) / (2*a)
        
        p1 = (x1 + t1*dx, y1 + t1*dy)
        p2 = (x1 + t2*dx, y1 + t2*dy)
        
        return [p1, p2]

    @staticmethod
    def init_obtuse_triangle(apex_idx: int = 0, scale: float = 1.35) -> List[Tuple[float, float]]:
        """Initialize an obtuse triangle"""
        coords = [
            (0.0 * scale, 0.0 * scale),
            (1.0 * scale, 0.0 * scale),
            (-0.3 * scale, 0.5 * scale),
        ]
        if apex_idx != 0:
            coords = [coords[apex_idx]] + [coords[i] for i in range(3) if i != apex_idx]
        return coords

    @staticmethod
    def add_noise(coords: List[Tuple[float, float]], noise_scale: float = 0.05) -> List[Tuple[float, float]]:
        import random
        return [
            (x + random.uniform(-noise_scale, noise_scale),
             y + random.uniform(-noise_scale, noise_scale))
            for x, y in coords
        ]
>>>>>>> minh-re
