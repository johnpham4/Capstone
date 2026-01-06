"""PyTorch optimizer for geometry constraints"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from primitives import Point, Line
from constraint import Constraint


class GeometryOptimizer(nn.Module):
    """PyTorch-based optimizer for geometric constraints"""

    def __init__(self, points: List[Point], initial_coords: Dict[Point, Tuple[float, float]]):
        super().__init__()

        self.point_names = points
        self.n_points = len(points)

        # Initialize point coordinates as learnable parameters
        coords = []
        for p in points:
            if p in initial_coords:
                x, y = initial_coords[p]
                coords.extend([x, y])
            else:
                coords.extend([0.0, 0.0])

        self.coords = nn.Parameter(torch.tensor(coords, dtype=torch.float32))

    def get_point_coords(self, point_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get x, y coordinates for a point"""
        idx = point_idx * 2
        return self.coords[idx], self.coords[idx + 1]

    def get_all_coords(self) -> Dict[Point, Tuple[float, float]]:
        """Get all point coordinates as dictionary"""
        result = {}
        for i, p in enumerate(self.point_names):
            x, y = self.get_point_coords(i)
            result[p] = (x.item(), y.item())
        return result

    def distance(self, p1_idx: int, p2_idx: int) -> torch.Tensor:
        """Calculate distance between two points"""
        x1, y1 = self.get_point_coords(p1_idx)
        x2, y2 = self.get_point_coords(p2_idx)
        return torch.sqrt((x2 - x1)**2 + (y2 - y1)**2 + 1e-8)

    def distance_squared(self, p1_idx: int, p2_idx: int) -> torch.Tensor:
        """Calculate squared distance between two points"""
        x1, y1 = self.get_point_coords(p1_idx)
        x2, y2 = self.get_point_coords(p2_idx)
        return (x2 - x1)**2 + (y2 - y1)**2

    def dot_product(self, p1_idx: int, p2_idx: int, p3_idx: int, p4_idx: int) -> torch.Tensor:
        """Dot product of vectors (p1->p2) and (p3->p4)"""
        x1, y1 = self.get_point_coords(p1_idx)
        x2, y2 = self.get_point_coords(p2_idx)
        x3, y3 = self.get_point_coords(p3_idx)
        x4, y4 = self.get_point_coords(p4_idx)

        v1_x = x2 - x1
        v1_y = y2 - y1
        v2_x = x4 - x3
        v2_y = y4 - y3

        return v1_x * v2_x + v1_y * v2_y

    def cross_product_2d(self, p1_idx: int, p2_idx: int, p3_idx: int) -> torch.Tensor:
        """2D cross product for collinearity: (p2-p1) × (p3-p1)"""
        x1, y1 = self.get_point_coords(p1_idx)
        x2, y2 = self.get_point_coords(p2_idx)
        x3, y3 = self.get_point_coords(p3_idx)

        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)

    def angle_constraint(self, p1_idx: int, p2_idx: int, p3_idx: int) -> torch.Tensor:
        """Constraint for angle at p2 formed by p1-p2-p3 = 90 degrees"""
        # Dot product should be 0 for perpendicular vectors
        x1, y1 = self.get_point_coords(p1_idx)
        x2, y2 = self.get_point_coords(p2_idx)
        x3, y3 = self.get_point_coords(p3_idx)

        v1_x = x1 - x2
        v1_y = y1 - y2
        v2_x = x3 - x2
        v2_y = y3 - y2

        return v1_x * v2_x + v1_y * v2_y

    def compute_constraint_loss(self, constraints: List[Constraint],
                               point_map: Dict[Point, int],
                               line_coords: Dict[Line, Tuple[int, int]],
                               on_segment_constraints: Dict[Point, Tuple[Point, Point]] = None) -> torch.Tensor:
        """Compute total loss from all constraints"""
        if on_segment_constraints is None:
            on_segment_constraints = {}

        total_loss = torch.tensor(0.0, dtype=torch.float32)

        for cons in constraints:
            pred = cons.pred
            args = cons.args

            try:
                if pred == "cong":  # Equal distances
                    p1_idx = point_map[args[0]]
                    p2_idx = point_map[args[1]]
                    p3_idx = point_map[args[2]]
                    p4_idx = point_map[args[3]]

                    d1 = self.distance_squared(p1_idx, p2_idx)
                    d2 = self.distance_squared(p3_idx, p4_idx)
                    total_loss = total_loss + (d1 - d2) ** 2

                elif pred == "midp":  # M is midpoint of AB
                    m_idx = point_map[args[0]]
                    a_idx = point_map[args[1]]
                    b_idx = point_map[args[2]]

                    xm, ym = self.get_point_coords(m_idx)
                    xa, ya = self.get_point_coords(a_idx)
                    xb, yb = self.get_point_coords(b_idx)

                    loss_x = (xm - (xa + xb) / 2) ** 2
                    loss_y = (ym - (ya + yb) / 2) ** 2
                    total_loss = total_loss + loss_x + loss_y

                elif pred == "coll":  # Three points collinear
                    p1_idx = point_map[args[0]]
                    p2_idx = point_map[args[1]]
                    p3_idx = point_map[args[2]]

                    cross = self.cross_product_2d(p1_idx, p2_idx, p3_idx)
                    total_loss = total_loss + cross ** 2

                elif pred == "right":  # Right angle at p2
                    p1_idx = point_map[args[0]]
                    p2_idx = point_map[args[1]]
                    p3_idx = point_map[args[2]]

                    angle_loss = self.angle_constraint(p1_idx, p2_idx, p3_idx)
                    total_loss = total_loss + angle_loss ** 2

                elif pred == "para":  # Parallel lines
                    # For parallel lines: slopes should be equal
                    # Line through p1-p2 parallel to line through p3-p4
                    # Cross product of direction vectors should be 0
                    if all(isinstance(arg, Point) for arg in args[:4]):
                        p1_idx = point_map[args[0]]
                        p2_idx = point_map[args[1]]
                        p3_idx = point_map[args[2]]
                        p4_idx = point_map[args[3]]

                        x1, y1 = self.get_point_coords(p1_idx)
                        x2, y2 = self.get_point_coords(p2_idx)
                        x3, y3 = self.get_point_coords(p3_idx)
                        x4, y4 = self.get_point_coords(p4_idx)

                        # Direction vectors
                        v1_x = x2 - x1
                        v1_y = y2 - y1
                        v2_x = x4 - x3
                        v2_y = y4 - y3

                        # Cross product should be 0
                        cross = v1_x * v2_y - v1_y * v2_x
                        total_loss = total_loss + cross ** 2

                elif pred == "perp":  # Perpendicular lines
                    # Dot product of direction vectors should be 0
                    if all(isinstance(arg, Point) for arg in args[:4]):
                        p1_idx = point_map[args[0]]
                        p2_idx = point_map[args[1]]
                        p3_idx = point_map[args[2]]
                        p4_idx = point_map[args[3]]

                        dot = self.dot_product(p1_idx, p2_idx, p3_idx, p4_idx)
                        total_loss = total_loss + dot ** 2

                elif pred == "eqangle":  # Equal angles
                    # Angle between p1-p2-p3 equals angle between p4-p5-p6
                    p1_idx = point_map[args[0]]
                    p2_idx = point_map[args[1]]
                    p3_idx = point_map[args[2]]
                    p4_idx = point_map[args[3]]
                    p5_idx = point_map[args[4]]
                    p6_idx = point_map[args[5]]

                    # Calculate cos of both angles
                    # cos(angle) = dot / (len1 * len2)
                    x1, y1 = self.get_point_coords(p1_idx)
                    x2, y2 = self.get_point_coords(p2_idx)
                    x3, y3 = self.get_point_coords(p3_idx)

                    v1_x = x1 - x2
                    v1_y = y1 - y2
                    v2_x = x3 - x2
                    v2_y = y3 - y2

                    dot1 = v1_x * v2_x + v1_y * v2_y
                    len1 = torch.sqrt(v1_x**2 + v1_y**2 + 1e-8)
                    len2 = torch.sqrt(v2_x**2 + v2_y**2 + 1e-8)
                    cos1 = dot1 / (len1 * len2 + 1e-8)

                    x4, y4 = self.get_point_coords(p4_idx)
                    x5, y5 = self.get_point_coords(p5_idx)
                    x6, y6 = self.get_point_coords(p6_idx)

                    v3_x = x4 - x5
                    v3_y = y4 - y5
                    v4_x = x6 - x5
                    v4_y = y6 - y5

                    dot2 = v3_x * v4_x + v3_y * v4_y
                    len3 = torch.sqrt(v3_x**2 + v3_y**2 + 1e-8)
                    len4 = torch.sqrt(v4_x**2 + v4_y**2 + 1e-8)
                    cos2 = dot2 / (len3 * len4 + 1e-8)

                    total_loss = total_loss + (cos1 - cos2) ** 2

            except KeyError:
                # Point not in map, skip this constraint
                continue

        # Add on-segment constraints (QUAN TRỌNG để giữ D, E trên đoạn thẳng)
        for point, (p1, p2) in on_segment_constraints.items():
            if point not in point_map or p1 not in point_map or p2 not in point_map:
                continue

            pt_idx = point_map[point]
            p1_idx = point_map[p1]
            p2_idx = point_map[p2]

            # Point must be on line segment: collinear + between p1 and p2
            # 1. Collinearity: cross product = 0
            cross = self.cross_product_2d(p1_idx, pt_idx, p2_idx)
            total_loss = total_loss + 10.0 * cross ** 2  # Strong penalty

            # 2. Between: 0 <= t <= 1 where point = p1 + t*(p2-p1)
            x_pt, y_pt = self.get_point_coords(pt_idx)
            x1, y1 = self.get_point_coords(p1_idx)
            x2, y2 = self.get_point_coords(p2_idx)

            # Calculate t from x coordinate
            dx = x2 - x1
            dy = y2 - y1

            if torch.abs(dx) > torch.abs(dy):
                t = (x_pt - x1) / (dx + 1e-8)
            else:
                t = (y_pt - y1) / (dy + 1e-8)

            # Penalize if t < 0 or t > 1
            if t.item() < 0:
                total_loss = total_loss + 100.0 * t ** 2
            elif t.item() > 1:
                total_loss = total_loss + 100.0 * (t - 1) ** 2

        # Add regularization to keep points spread out
        for i in range(len(self.point_names)):
            for j in range(i + 1, len(self.point_names)):
                d_sq = self.distance_squared(i, j)
                # Penalize if points too close
                if d_sq < 0.01:
                    total_loss = total_loss + 10.0 * (0.1 - d_sq) ** 2

        return total_loss

    def optimize(self, constraints: List[Constraint],
                point_map: Dict[Point, int],
                line_coords: Dict[Line, Tuple[int, int]] = None,
                on_segment_constraints: Dict[Point, Tuple[Point, Point]] = None,
                n_iterations: int = 1000,
                lr: float = 0.01) -> Dict[Point, Tuple[float, float]]:
        """Run optimization to satisfy constraints"""
        if line_coords is None:
            line_coords = {}
        if on_segment_constraints is None:
            on_segment_constraints = {}

        optimizer = torch.optim.Adam([self.coords], lr=lr)

        for iteration in range(n_iterations):
            optimizer.zero_grad()

            loss = self.compute_constraint_loss(constraints, point_map, line_coords, on_segment_constraints)

            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: loss = {loss.item():.6f}")

        return self.get_all_coords()
