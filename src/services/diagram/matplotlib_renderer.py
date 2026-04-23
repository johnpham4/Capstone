
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from loguru import logger
import matplotlib.patches as patches

from .model.entities import Diagram, GeometricPoint
from .model.types import QuadrilateralType


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

        # Size of right angle symbol
        size = min(len1, len2) * 0.15

        # Draw small square
        corner1 = (vertex.x + v1x * size, vertex.y + v1y * size)
        corner2 = (vertex.x + v1x * size + v2x * size, vertex.y + v1y * size + v2y * size)
        corner3 = (vertex.x + v2x * size, vertex.y + v2y * size)

        ax.plot([corner1[0], corner2[0], corner3[0]],
                [corner1[1], corner2[1], corner3[1]],
                color='green', linewidth=1.5)

    def _find_segment_intersection(self, p1: GeometricPoint, p2: GeometricPoint,
                                    p3: GeometricPoint, p4: GeometricPoint):
        """
        Tìm giao điểm của 2 đoạn thẳng: seg1=(p1,p2) và seg2=(p3,p4)
        Trả về (x, y) nếu có giao điểm, None nếu không giao nhau
        """
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = p3.x, p3.y
        x4, y4 = p4.x, p4.y

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        # Kiểm tra song song
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        # Kiểm tra giao điểm có nằm trên cả 2 đoạn thẳng không
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)

        return None

    def _draw_angle_arc(self, ax, vertex: GeometricPoint, p1: GeometricPoint, p2: GeometricPoint, num_arcs: int = 1, radius: float = 0.12, draw_tick: bool = False):
        """
        vertex: GeometricPoint - đỉnh góc
        p1, p2: GeometricPoint - 2 điểm tạo thành góc
        num_arcs: số arc (1 arc, 2 arcs, 3 arcs để phân biệt)
        draw_tick: có vẽ dấu gạch nhỏ trên arc không (để chỉ góc bằng nhau)
        draw_tick: có vẽ dấu gạch nhỏ trên arc không (để chỉ góc bằng nhau)
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

        # Vẽ dấu gạch nhỏ ở giữa arc
        if draw_tick:
            # Tính góc giữa của arc
            mid_angle = (angle1 + angle2) / 2
            mid_rad = mid_angle * np.pi / 180

            # Vị trí giữa arc (sử dụng radius của arc ngoài cùng)
            r_mid = radius + (num_arcs - 1) * 0.05
            mid_x = vertex.x + r_mid * np.cos(mid_rad)
            mid_y = vertex.y + r_mid * np.sin(mid_rad)

            # Vector RADIAL (từ tâm ra ngoài) - cắt vuông góc qua arc
            tick_dx = np.cos(mid_rad)
            tick_dy = np.sin(mid_rad)

            # Độ dài dấu gạch (cắt qua arc) - giảm từ 0.04 xuống 0.025
            tick_length = 0.025

            ax.plot(
                [mid_x - tick_dx * tick_length, mid_x + tick_dx * tick_length],
                [mid_y - tick_dy * tick_length, mid_y + tick_dy * tick_length],
                'b-',
                linewidth=1.5
            )

    def _draw_angle_measure(self, ax, vertex: GeometricPoint, p1: GeometricPoint, p2: GeometricPoint, angle_degrees: float, radius: float = 0.16):
        """Vẽ số đo góc tại đỉnh giữa hai tia p1-vertex và p2-vertex."""
        v1x = p1.x - vertex.x
        v1y = p1.y - vertex.y
        v2x = p2.x - vertex.x
        v2y = p2.y - vertex.y

        v1_norm = np.sqrt(v1x**2 + v1y**2)
        v2_norm = np.sqrt(v2x**2 + v2y**2)
        if v1_norm < 1e-8 or v2_norm < 1e-8:
            return

        v1x, v1y = v1x / v1_norm, v1y / v1_norm
        v2x, v2y = v2x / v2_norm, v2y / v2_norm

        angle1 = np.arctan2(v1y, v1x)
        angle2 = np.arctan2(v2y, v2x)
        mid_angle = (angle1 + angle2) / 2

        angle_diff = angle2 - angle1
        if angle_diff > np.pi:
            mid_angle += np.pi
        elif angle_diff < -np.pi:
            mid_angle -= np.pi

        text_radius = radius * 0.85
        text_x = vertex.x + text_radius * np.cos(mid_angle)
        text_y = vertex.y + text_radius * np.sin(mid_angle)

        ax.text(
            text_x,
            text_y,
            f"{int(angle_degrees)}°",
            fontsize=16,
            ha='center',
            va='center',
            color='blue',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=0.8),
        )


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
            logger.warning("No diagram provided, creating empty figure")
            # Create empty figure instead of raising error
            fig, ax = plt.subplots(figsize=(20, 20))
            ax.text(0.5, 0.5, 'No diagram data',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes,
                   fontsize=20,
                   color='gray')
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_aspect('equal', 'box')
            ax.axis('off')

            if save:
                output_path = Path(filename)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
                logger.info(f"Saved empty diagram to {output_path}")
            if show:
                plt.show()
            return fig, ax

        fig, ax = plt.subplots(figsize=(20, 20))

        # Vertices with explicit angle-measure labels should show a single angle arc.
        measured_vertex_ids = set()
        if hasattr(self.diagram, 'angle_measures') and self.diagram.angle_measures:
            measured_vertex_ids = {id(angle_data['vertex']) for angle_data in self.diagram.angle_measures}

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
                    # Avoid duplicate markers where a numeric angle is already shown.
                    if id(pts[idx1]) not in measured_vertex_ids:
                        self._draw_angle_arc(ax, pts[idx1],
                                            pts[(idx1-1)%3],
                                            pts[(idx1+1)%3],
                                            num_arcs=1,
                                            draw_tick=True)

                    if id(pts[idx2]) not in measured_vertex_ids:
                        self._draw_angle_arc(ax, pts[idx2],
                                            pts[(idx2-1)%3],
                                            pts[(idx2+1)%3],
                                            num_arcs=1,
                                            draw_tick=True)

        # Draw quadrilaterals
        for quad in self.diagram.quadrilaterals:
            points = quad['points']
            quad_type = quad.get('type', QuadrilateralType.GENERAL)

            # Determine quad_type as string for comparison
            if isinstance(quad_type, str):
                quad_type_str = quad_type.lower()
            else:
                quad_type_str = str(quad_type).split('.')[-1].lower()

            # Draw edges
            xs = [p.x for p in points] + [points[0].x]
            ys = [p.y for p in points] + [points[0].y]
            ax.plot(xs, ys, 'k-', linewidth=1.5)

            # SQUARE: All right angles + all sides equal (1 tick each)
            if quad_type_str == 'square':
                # Draw right angle markers at all 4 corners
            # SQUARE: All right angles + all sides equal (1 tick each)
            if quad_type_str == 'square':
                # Draw right angle markers at all 4 corners
                for i in range(4):
                    vertex = points[i]
                    p1 = points[(i - 1) % 4]
                    p2 = points[(i + 1) % 4]
                    self._draw_right_angle_symbol(ax, vertex, p1, p2)

                # Draw 1 tick on all 4 sides
                for i in range(4):
                    p1 = points[i]
                    p2 = points[(i + 1) % 4]
                    self._draw_tick_marks(ax, p1, p2, 1)

            # RECTANGLE: All right angles + opposite sides equal
            elif quad_type_str == 'rectangle':
                # Draw right angle markers at all 4 corners
                for i in range(4):
                    vertex = points[i]
                    p1 = points[(i - 1) % 4]
                    p2 = points[(i + 1) % 4]
                    self._draw_right_angle_symbol(ax, vertex, p1, p2)

                # Draw different ticks for opposite sides: 1 tick on AB/CD, 2 ticks on BC/DA
                for i in range(4):
                    p1 = points[i]
                    p2 = points[(i + 1) % 4]
                    num_ticks = 1 if i % 2 == 0 else 2
                    self._draw_tick_marks(ax, p1, p2, num_ticks)

            # RHOMBUS: All sides equal + diagonals perpendicular
            elif quad_type_str == 'rhombus':
                # Draw 1 tick on all 4 sides (all equal)
                for i in range(4):
                    p1 = points[i]
                    p2 = points[(i + 1) % 4]
                    self._draw_tick_marks(ax, p1, p2, 1)

                # Keep rhombus rendering clean: no implicit diagonal/right-angle
                # decorations unless explicitly requested by DSL constraints.

            # QUADRILATERAL (generic): No special markings
            # Just the outline is already drawn above

        # Draw circles (radius calculated in optimizer)
        for center, info in self.diagram.circles:
            logger.info(f"Drawing circle at {center.name} ({center.x:.4f}, {center.y:.4f}), info: {info}")
            logger.info(f"Drawing circle at {center.name} ({center.x:.4f}, {center.y:.4f}), info: {info}")
            if isinstance(info, dict):
                radius = info.get('radius', 0.5)  # Use calculated radius from optimizer
                logger.info(f"  Radius: {radius}, Type: {info.get('type')}")

                # Color by circle type
                color_map = {
                    'incircle': 'blue',
                    'circumcircle': 'green',
                    'positioned': 'black'
                }
                color = color_map.get(info.get('type', 'positioned'), 'black')

                circle = plt.Circle((center.x, center.y), radius, fill=False,
                                  edgecolor=color, linewidth=1.0)
                ax.add_patch(circle)
                logger.info(f"  Circle added to plot")
            else:
                # Fallback for old format
                circle = plt.Circle((center.x, center.y), info, fill=False,
                                  edgecolor='black', linewidth=1.0)
                ax.add_patch(circle)

        # 4. Draw Segments (Auxiliary - Dashed)
        for p1, p2, color in self.diagram.segments:
            ax.plot([p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=1.0, linestyle='--', alpha=0.7)

        # 4.5. Draw Perpendicular Markers
        if hasattr(self.diagram, 'perpendiculars') and self.diagram.perpendiculars:
            for perp_tuple in self.diagram.perpendiculars:
                # perp_tuple = (name1, name2, name3, name4) - string names from optimizer
                if len(perp_tuple) == 4:
                    name1, name2, name3, name4 = perp_tuple

                    # Convert names to GeometricPoint objects
                    if all(name in self.diagram.points for name in [name1, name2, name3, name4]):
                        p1 = self.diagram.points[name1]
                        p2 = self.diagram.points[name2]
                        p3 = self.diagram.points[name3]
                        p4 = self.diagram.points[name4]

                        # Tìm giao điểm
                        intersection = self._find_segment_intersection(p1, p2, p3, p4)

                        if intersection:
                            # Tạo temporary point cho giao điểm
                            class TempPoint:
                                def __init__(self, x, y):
                                    self.x = x
                                    self.y = y

                            intersection_pt = TempPoint(intersection[0], intersection[1])

                            # Sử dụng 2 điểm trên mỗi đoạn thẳng để xác định hướng
                            self._draw_right_angle_symbol(ax, intersection_pt, p1, p3)


        # 5. Draw Lines
        for line_name, line_data in self.diagram.lines.items():
            p1, p2 = line_data
            dx = p2.x - p1.x
            dy = p2.y - p1.y
            length = np.sqrt(dx**2 + dy**2)
            print(f"Drawing line '{line_name}': ({p1.x:.4f}, {p1.y:.4f}) to ({p2.x:.4f}, {p2.y:.4f})")

            if length > 0:
                dx /= length
                dy /= length
                extend_factor = 5.0

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

        # 5.5 Draw right angle for all perpendicular constraints (lines and segments)
        if hasattr(self.diagram, 'perpendicular_lines') and self.diagram.perpendicular_lines:
            for perp in self.diagram.perpendicular_lines:
                # perp = (intersection_pt, dir1_pt, dir2_pt)
                vertex, p1, p2 = perp
                self._draw_right_angle_symbol(ax, vertex, p1, p2)

        # 6. Draw Points and Labels
        if self.diagram.points:
            cx = 0.0
            cy = 0.0

            for name, p in self.diagram.points.items():
                if name.endswith('_aux'):
                    continue
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
                    fontsize=24,
                    fontsize=24,
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
                    if id(vertex) in measured_vertex_ids:
                        continue
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

                # Draw arc for angle 1 (p1-vertex-p2) với dấu gạch
                self._draw_angle_arc(ax, angle1['vertex'],
                                   angle1['p1'],
                                   angle1['p2'],
                                   num_arcs=1,
                                   radius=0.12,
                                   draw_tick=True)

                # Draw arc for angle 2 với dấu gạch
                self._draw_angle_arc(ax, angle2['vertex'],
                                   angle2['p1'],
                                   angle2['p2'],
                                   num_arcs=1,
                                   radius=0.12,
                                   draw_tick=True)

        # Issue 2: Draw angle bisector arcs (2 arcs for equal angles)
        if hasattr(self.diagram, 'angle_bisectors_metadata') and self.diagram.angle_bisectors_metadata:
            for bisector_data in self.diagram.angle_bisectors_metadata:
                vertex_name = bisector_data['vertex']
                bisector_pt_name = bisector_data['bisector_point']
                angle_pts = bisector_data['angle_points']  # [B, A, C] where A is vertex

                vertex = self.diagram.get_point(vertex_name)
                bisector_pt = self.diagram.get_point(bisector_pt_name)
                p1 = self.diagram.get_point(angle_pts[0])  # B
                p2 = self.diagram.get_point(angle_pts[2])  # C

                if vertex and bisector_pt and p1 and p2:
                    # Arc 1: angle from p1 to bisector (góc BAM)
                    self._draw_angle_arc(ax, vertex, p1, bisector_pt,
                                       num_arcs=1, radius=0.12, draw_tick=True)
                    # Arc 2: angle from bisector to p2 (góc MAC)
                    self._draw_angle_arc(ax, vertex, bisector_pt, p2,
                                       num_arcs=1, radius=0.12, draw_tick=True)

        # Draw angle measures (display degree values)
        if hasattr(self.diagram, 'angle_measures') and self.diagram.angle_measures:
            for angle_data in self.diagram.angle_measures:
                degrees = float(angle_data['degrees'])

                # For right angles, show only the green square marker (no blue arc, no degree text).
                if abs(degrees - 90.0) < 1e-6:
                    self._draw_right_angle_symbol(
                        ax,
                        angle_data['vertex'],
                        angle_data['p1'],
                        angle_data['p2']
                    )
                    continue

                # Non-right angles: draw blue arc and blue degree value.
                self._draw_angle_arc(
                    ax,
                    angle_data['vertex'],
                    angle_data['p1'],
                    angle_data['p2'],
                    num_arcs=1,
                    radius=0.12,
                    draw_tick=False
                )
                self._draw_angle_measure(
                    ax,
                    angle_data['vertex'],
                    angle_data['p1'],
                    angle_data['p2'],
                    degrees,
                    radius=0.22
                )


        # Configure axes
        ax.set_aspect('equal')

        # Calculate bounding box for ALL elements (points + circles)
        if self.diagram.points:
            all_x = [p.x for p in self.diagram.points.values()]
            all_y = [p.y for p in self.diagram.points.values()]

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # Extend bounding box to include circles
            if self.diagram.circles:
                for center, info in self.diagram.circles:
                    if isinstance(info, dict):
                        radius = info.get('radius', 0.5)
                    else:
                        radius = info

                    min_x = min(min_x, center.x - radius)
                    max_x = max(max_x, center.x + radius)
                    min_y = min(min_y, center.y - radius)
                    max_y = max(max_y, center.y + radius)

            # Calculate true center of the entire diagram
            actual_center_x = (min_x + max_x) / 2
            actual_center_y = (min_y + max_y) / 2

            # Calculate zoom factor with padding
            x_range = max_x - min_x
            y_range = max_y - min_y
            max_range = max(x_range, y_range) / 2
            padding = max(0.15, max_range * 0.25)
            zoom_factor = max(0.55, max_range + padding)

            # Center the view on the actual center of all elements
            ax.set_xlim(actual_center_x - zoom_factor, actual_center_x + zoom_factor)
            ax.set_ylim(actual_center_y - zoom_factor, actual_center_y + zoom_factor)
        else:
            # Fallback if no points
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)

        ax.axis('off')

        # Save if requested
        if save:
            output_path = Path(filename)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
                        pad_inches=0.3, facecolor='white', edgecolor='none')
            logger.info(f"Diagram saved to: {output_path}")


        if show:
            plt.show()
        elif not save:
            plt.close(fig)

        return fig, ax

