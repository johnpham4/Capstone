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

    @staticmethod
    def init_equilateral_triangle(scale: float = 1.0) -> List[Tuple[float, float]]:
        height = math.sqrt(3) / 2 * scale
        return [
            (0.0, 2 * height / 3),           # Top
            (-0.5 * scale, -height / 3),     # Bottom left
            (0.5 * scale, -height / 3)       # Bottom right
        ]

    @staticmethod
    def init_scalene_triangle(scale: float = 2.0):
        return [
            (0.0, 0.8 * scale),        # A: top
            (-0.7 * scale, -0.4 * scale),  # B: left
            (0.9 * scale, -0.2 * scale),   # C: right (không đối xứng)
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
    def add_noise(coords: List[Tuple[float, float]], noise_scale: float = 0.05) -> List[Tuple[float, float]]:
        import random
        return [
            (x + random.uniform(-noise_scale, noise_scale),
             y + random.uniform(-noise_scale, noise_scale))
            for x, y in coords
        ]
