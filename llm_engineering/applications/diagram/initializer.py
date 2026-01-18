import math
from typing import List, Tuple


class Initializer:

    @staticmethod
    def init_isoceles_triangle(apex_idx: int = 0, scale: float = 1.0) -> List[Tuple[float, float]]:
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
    def init_right_triangle(right_angle_idx: int = 0, scale: float = 1.0) -> List[Tuple[float, float]]:
        base_coords = [
            (0.0, 0.0),
            (0.8 * scale, 0.0),      # Along x-axis
            (0.0, 0.8 * scale)       # Along y-axis
        ]

        if right_angle_idx == 0:
            return base_coords
        elif right_angle_idx == 1:
            return [base_coords[1], base_coords[0], base_coords[2]]
        else:
            return [base_coords[1], base_coords[2], base_coords[0]]

    def init_equal_angle_triangle(self, scale: float = 1.0) -> List[Tuple[float, float]]:
        """Initialize triangle with two equal angles"""
        # Isoceles triangle with apex at top
        height = 0.8 * scale
        half_base = 0.6 * scale
        return [
            (0.0, height),          # Apex
            (-half_base, 0.0),     # Base left
            (half_base, 0.0)       # Base right
        ]
        
        
    @staticmethod
    def init_equilateral_triangle(scale: float = 1.0) -> List[Tuple[float, float]]:
        height = math.sqrt(3) / 2 * scale
        return [
            (0.0, 2 * height / 3),           # Top
            (-0.5 * scale, -height / 3),     # Bottom left
            (0.5 * scale, -height / 3)       # Bottom right
        ]

    @staticmethod
    def init_scalene_triangle(scale: float = 1.0) -> List[Tuple[float, float]]:
        """Scalene triangle with all sides different"""
        return [
            (-0.6 * scale, -0.3 * scale),    # Bottom left
            (0.5 * scale, -0.2 * scale),     # Bottom right (shorter base)
            (0.1 * scale, 0.7 * scale)       # Top (offset to left)
        ]

    @staticmethod
    def init_right_isoceles_triangle(right_angle_idx: int = 0, scale: float = 1.0) -> List[Tuple[float, float]]:
        leg_length = 0.7 * scale
        base_coords = [
            (0.0, 0.0),                    # Right angle
            (leg_length, 0.0),             # Horizontal leg
            (0.0, leg_length)              # Vertical leg
        ]

        if right_angle_idx == 0:
            return base_coords
        elif right_angle_idx == 1:
            return [base_coords[1], base_coords[0], base_coords[2]]
        else:
            return [base_coords[1], base_coords[2], base_coords[0]]

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
        return Initializer.init_rectangle(side, side)

    @staticmethod
    def init_scalene_quadrilateral(scale: float = 1.0) -> List[Tuple[float, float]]:
        return [
            (-0.6 * scale, -0.3 * scale),    # Bottom left
            (0.5 * scale, -0.2 * scale),     # Bottom right (shorter base)
            (0.1 * scale, 0.7 * scale),      # Top right (offset to left)
            (-0.4 * scale, 0.4 * scale)      # Top left
        ]

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
    def add_noise(coords: List[Tuple[float, float]], noise_scale: float = 0.05) -> List[Tuple[float, float]]:
        import random
        return [
            (x + random.uniform(-noise_scale, noise_scale),
             y + random.uniform(-noise_scale, noise_scale))
            for x, y in coords
        ]
