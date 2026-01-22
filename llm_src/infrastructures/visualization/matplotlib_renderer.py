import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.patches as patches

from llm_src.domains.geometry import Diagram, GeometricPoint
from llm_src.domains.geometry.types import QuadrilateralType


class MatplotlibDiagramRenderer:

    def __init__(self, diagram: Diagram = None):
        self.diagram = diagram

    def _draw_tick_marks(self, ax, p1: GeometricPoint, p2: GeometricPoint, num_ticks: int) -> None:
        mx, my = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
        dx, dy = p2.x - p1.x, p2.y - p1.y
        length = np.sqrt(dx**2 + dy**2)

        # Normal vector (perpendicular)
        nx, ny = -dy / length, dx / length

        tick_size = length * 0.05
        tick_spacing = length * 0.03

        # Draw multiple ticks
        for i in range(num_ticks):
            offset = (i - (num_ticks - 1) / 2) * tick_spacing
            # Perpendicular offset
            tx, ty = mx + dx / length * offset, my + dy / length * offset

            ax.plot(
                [tx - nx * tick_size, tx + nx * tick_size],
                [ty - ny * tick_size, ty + ny * tick_size],
                'k-',
                linewidth=1.5
            )

    def _draw_right_angle_symbol(self, ax, vertex: GeometricPoint, p1: GeometricPoint, p2: GeometricPoint) -> None:
        # Vectors from vertex to other two points
        v1x, v1y = p1.x - vertex.x, p1.y - vertex.y
        v2x, v2y = p2.x - vertex.x, p2.y - vertex.y

        # Normalize
        len1 = np.sqrt(v1x**2 + v1y**2)
        len2 = np.sqrt(v2x**2 + v2y**2)

        if len1 == 0 or len2 == 0:
            return

        v1x, v1y = v1x / len1, v1y / len1
        v2x, v2y = v2x / len2, v2y / len2

        # Size of right angle symbol (10% of shortest leg)
        size = min(len1, len2) * 0.1

        # Draw small square
        corner1 = (vertex.x + v1x * size, vertex.y + v1y * size)
        corner2 = (vertex.x + v1x * size + v2x * size, vertex.y + v1y * size + v2y * size)
        corner3 = (vertex.x + v2x * size, vertex.y + v2y * size)

        ax.plot([corner1[0], corner2[0], corner3[0]],
                [corner1[1], corner2[1], corner3[1]],
                'k-', linewidth=1.0)

    def _draw_angle_arc(self, ax, vertex: GeometricPoint, p1: GeometricPoint, p2: GeometricPoint, num_arcs: int = 1, radius: float = 0.12):
        """
        vertex: GeometricPoint - đỉnh góc
        p1, p2: GeometricPoint - 2 điểm tạo thành góc
        num_arcs: số arc (1 arc, 2 arcs, 3 arcs để phân biệt)
        """
        # Vector từ vertex đến 2 điểm
        v1x = p1.x - vertex.x
        v1y = p1.y - vertex.y
        v2x = p2.x - vertex.x
        v2y = p2.y - vertex.y

        # Normalize
        v1_norm = np.sqrt(v1x**2 + v1y**2)
        v2_norm = np.sqrt(v2x**2 + v2y**2)

        if v1_norm < 1e-8 or v2_norm < 1e-8:
            return  # Skip if points are too close

        v1x = v1x / v1_norm
        v1y = v1y / v1_norm
        v2x = v2x / v2_norm
        v2y = v2y / v2_norm

        # Tính góc (degrees)
        angle1 = np.arctan2(v1y, v1x) * 180 / np.pi
        angle2 = np.arctan2(v2y, v2x) * 180 / np.pi

        # Đảm bảo vẽ góc nhỏ hơn 180 độ
        if angle2 < angle1:
            angle1, angle2 = angle2, angle1
        if angle2 - angle1 > 180:
            angle1, angle2 = angle2, angle1 + 360

        # Vẽ nhiều arcs để phân biệt
        for i in range(num_arcs):
            r = radius + i * 0.05  # Mỗi arc cách nhau 0.05
            arc = patches.Arc((vertex.x, vertex.y), 2*r, 2*r,
                             angle=0,
                             theta1=angle1,
                             theta2=angle2,
                             color='blue',
                             linewidth=1.2)
            ax.add_patch(arc)


    def render(
        self,
        diagram: Diagram = None,
        show: bool = False,
        save: bool = True,
        filename: str = "diagram_output.png"
    ):

        if diagram:
            self.diagram = diagram

        if not self.diagram:
            raise ValueError("No diagram to render")

        fig, ax = plt.subplots(figsize=(8, 8))

        # 1. Draw Triangles (Solid Lines)
        for tri in self.diagram.triangles:
            p1, p2, p3 = tri[0], tri[1], tri[2]
            equal_sides = tri[3] if len(tri) > 3 else None
            right_angle_at = tri[4] if len(tri) > 4 else None
            equal_angles = tri[5] if len(tri) > 5 else None

            logger.info(f"Rendering triangle with equal_angles: {equal_angles} (tri length: {len(tri)})")

            xs = [p1.x, p2.x, p3.x, p1.x]
            ys = [p1.y, p2.y, p3.y, p1.y]
            ax.plot(xs, ys, 'k-', linewidth=1.5)

            # Define points array for all features
            pts = [p1, p2, p3]

            # Draw tick marks for equal sides
            if equal_sides:
                # Group equal sides by pairs
                sides_map = {}  # {(i,j): tick_num}
                for pair_idx, (i, j) in enumerate(equal_sides):
                    # Check if this pair already has ticks

                    found = False
                    for existing_sides, tick_num in sides_map.items():
                        if (i, j) in existing_sides or (j, i) in existing_sides:
                            found = True
                            break
                    if not found:
                        # Find all sides equal to this one
                        equal_group = [(i, j)]
                        for other_i, other_j in equal_sides:
                            if (other_i, other_j) != (i, j) and (other_j, other_i) != (i, j):
                                # Check if they share a vertex

                                if i in (other_i, other_j) or j in (other_i, other_j):
                                    equal_group.append((other_i, other_j))
                        sides_map[tuple(equal_group)] = len(sides_map) + 1


                drawn = set()
                for sides_group, num_ticks in sides_map.items():
                    for i, j in sides_group:
                        if (i, j) not in drawn and (j, i) not in drawn:
                            self._draw_tick_marks(ax, pts[i], pts[j], num_ticks)
                            drawn.add((i, j))

            # Draw right angle symbol
            if right_angle_at is not None:
                vertex = pts[right_angle_at]
                others = [pts[i] for i in range(3) if i != right_angle_at]
                self._draw_right_angle_symbol(ax, vertex, others[0], others[1])

            # Draw equal angles arcs
            if equal_angles:
                logger.info(f"Drawing equal angles arcs: {equal_angles}")
                for idx1, idx2 in equal_angles:
                    # Vẽ arc ở góc idx1
                    self._draw_angle_arc(ax, pts[idx1],
                                        pts[(idx1-1)%3],
                                        pts[(idx1+1)%3],
                                        num_arcs=1)

                    # Vẽ arc ở góc idx2 (cùng số arc)
                    self._draw_angle_arc(ax, pts[idx2],
                                        pts[(idx2-1)%3],
                                        pts[(idx2+1)%3],
                                        num_arcs=1)

        # Draw quadrilaterals
        for quad in self.diagram.quadrilaterals:
            points = quad['points']
            quad_type = quad.get('type', QuadrilateralType.GENERAL)

            # Draw edges
            xs = [p.x for p in points] + [points[0].x]
            ys = [p.y for p in points] + [points[0].y]
            ax.plot(xs, ys, 'k-', linewidth=1.5)

            # Draw right angle markers for all 4 corners (square/rectangle)
            if quad_type in [QuadrilateralType.SQUARE, QuadrilateralType.RECTANGLE]:
                for i in range(4):
                    vertex = points[i]
                    p1 = points[(i - 1) % 4]
                    p2 = points[(i + 1) % 4]
                    self._draw_right_angle_symbol(ax, vertex, p1, p2)

            # Draw equal side markings
            equal_sides = quad.get('equal_sides', [])
            if equal_sides:
                if quad_type == QuadrilateralType.SQUARE:
                    # All 4 sides equal - draw same number of ticks on all sides
                    for i in range(4):
                        p1 = points[i]
                        p2 = points[(i + 1) % 4]
                        self._draw_tick_marks(ax, p1, p2, 1)
                elif quad_type == QuadrilateralType.RECTANGLE:
                    # Opposite sides equal - draw 1 tick on AB/CD, 2 ticks on BC/DA
                    for i in range(4):
                        p1 = points[i]
                        p2 = points[(i + 1) % 4]
                        num_ticks = 1 if i % 2 == 0 else 2
                        self._draw_tick_marks(ax, p1, p2, num_ticks)

        # Draw circles
        for center, info in self.diagram.circles:
            if isinstance(info, dict):
                if info['type'] == 'incircle':
                    # Calculate incircle radius from triangle
                    tri_points = [self.diagram.points[name] for name in info['triangle'] if name in self.diagram.points]
                    if len(tri_points) == 3:
                        # Use distance from center to one side
                        p1, p2 = tri_points[0], tri_points[1]
                        # Distance formula from point to line

                        num = abs((p2.y - p1.y) * center.x - (p2.x - p1.x) * center.y + p2.x * p1.y - p2.y * p1.x)
                        den = np.sqrt((p2.y - p1.y)**2 + (p2.x - p1.x)**2)
                        radius = num / den if den > 0 else 0.1
                        circle = plt.Circle((center.x, center.y), radius, fill=False, edgecolor='blue', linewidth=1.0)
                        ax.add_patch(circle)
                elif info['type'] == 'circumcircle':

                    tri_points = [self.diagram.points[name] for name in info['triangle'] if name in self.diagram.points]
                    if len(tri_points) == 3:
                        radius = center.distance_to(tri_points[0])
                        circle = plt.Circle((center.x, center.y), radius, fill=False, edgecolor='green', linewidth=1.0)
                        ax.add_patch(circle)
            else:

                circle = plt.Circle((center.x, center.y), info, fill=False, edgecolor='black', linewidth=1.0)
                ax.add_patch(circle)

        # 4. Draw Segments (Auxiliary - Dashed)
        for p1, p2, color in self.diagram.segments:
            ax.plot([p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=1.0, linestyle='--', alpha=0.7)

        # 5. Draw Lines
        for line_name, line_data in self.diagram.lines.items():
            p1, p2 = line_data
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = np.sqrt(dx**2 + dy**2)

            if length > 0:
            # Normalize direction
                dx /= length
                dy /= length
                extend_factor = 2.0  # Extend line in both directions

                start_x = p1.x - dx * extend_factor
                start_y = p1.y - dy * extend_factor
                end_x = p2.x + dx * extend_factor
                end_y = p2.y + dy * extend_factor


                ax.plot([start_x, end_x], [start_y, end_y],
                       color='blue', linewidth=1.0, linestyle='-', alpha=0.6)

                arrow_size = 0.15
                ax.annotate('', xy=(end_x, end_y), xytext=(end_x - dx * arrow_size, end_y - dy * arrow_size),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=1.0, alpha=0.6))
                ax.annotate('', xy=(start_x, start_y), xytext=(start_x + dx * arrow_size, start_y + dy * arrow_size),
                           arrowprops=dict(arrowstyle='->', color='blue', lw=1.0, alpha=0.6))

        # 6. Draw Points and Labels
        if self.diagram.points:
            cx = sum(p.x for p in self.diagram.points.values()) / len(self.diagram.points)
            cy = sum(p.y for p in self.diagram.points.values()) / len(self.diagram.points)

            for name, p in self.diagram.points.items():
                ax.plot(p.x, p.y, 'ko', markersize=4)

                dx, dy = p.x - cx, p.y - cy
                dist = np.sqrt(dx**2 + dy**2)

                if dist > 0:
                    ox, oy = dx / dist * 12, dy / dist * 12
                else:
                    ox, oy = 8, 8

                ax.annotate(
                    name, (p.x, p.y),
                    xytext=(ox, oy),
                    textcoords='offset points',
                    fontsize=10,
                    fontweight='bold'
                )

        # Draw angle bisectors
        if hasattr(self.diagram, 'angle_bisectors') and self.diagram.angle_bisectors:
            for bisector_data in self.diagram.angle_bisectors:
                vertex = bisector_data['vertex']
                bisector_point = bisector_data['point']

                # Vẽ đoạn thẳng từ vertex đến bisector_point (đường màu xanh lá nét đứt)
                ax.plot([vertex.x, bisector_point.x],
                    [vertex.y, bisector_point.y],
                    'g--', linewidth=1.5, alpha=0.7)

                # Vẽ ký hiệu 2 góc bằng nhau (tương tự equal_angles)
                angle_points = bisector_data.get('angle_points', [])
                if len(angle_points) >= 3:
                    # Lấy 2 điểm tạo góc: angle_points = [A, B, C] (A là đỉnh, góc BAC bị chia)
                    p1 = self.diagram.points.get(angle_points[1])  # B
                    p2 = self.diagram.points.get(angle_points[2])  # C

                    if p1 and p2:
                        # Vẽ arc ở 2 bên của phân giác để chỉ 2 góc bằng nhau
                        self._draw_angle_arc(ax, vertex, p1, bisector_point, num_arcs=1, radius=0.08)
                        self._draw_angle_arc(ax, vertex, bisector_point, p2, num_arcs=1, radius=0.08)

        # Draw angle-equal assertions (angle ABC = angle DEF)
        if hasattr(self.diagram, 'angle_equal_assertions') and self.diagram.angle_equal_assertions:
            for assertion in self.diagram.angle_equal_assertions:
                angle1 = assertion['angle1']
                angle2 = assertion['angle2']

                # Draw arc for angle 1 (p1-vertex-p2)
                self._draw_angle_arc(ax, angle1['vertex'],
                                   angle1['p1'],
                                   angle1['p2'],
                                   num_arcs=1,
                                   radius=0.12)

                # Draw arc for angle 2
                self._draw_angle_arc(ax, angle2['vertex'],
                                   angle2['p1'],
                                   angle2['p2'],
                                   num_arcs=1,
                                   radius=0.12)

        # Configure axes
        ax.set_aspect('equal')
        ax.axis('off')

        # Save if requested

        if save:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), bbox_inches='tight', dpi=150)
            logger.info(f"Diagram saved to: {output_path}")


        if show:
            plt.show()
        elif not save:
            plt.close(fig)

        return fig, ax
