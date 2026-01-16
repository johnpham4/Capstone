import matplotlib.pyplot as plt
import numpy as np

from loguru import logger

class Diagram:
    def __init__(self):
        self.points = {}
        self.triangles = []
        self.segments = []
        self.lines = {}
        self.equal_segments = []

    def add_point(self, name, point):
        self.points[name] = point

    def add_triangle(self, p1, p2, p3, equal_sides=None):
        """equal_sides: list các tuple chỉ cặp đỉnh có cạnh bằng nhau, vd [(0,1), (0,2)] cho tam giác cân tại đỉnh 0"""
        self.triangles.append((p1, p2, p3, equal_sides))
    
    def add_quadrilateral(self, p1, p2, p3, p4, draw_diagonals=False):
        """Thêm tứ giác ABCD bằng cách vẽ 4 cạnh"""
        self.add_segment(p1, p2)
        self.add_segment(p2, p3)
        self.add_segment(p3, p4)
        self.add_segment(p4, p1)
        
        # if draw_diagonals:
        #     self.lines.append((p1, p3))  
        #     self.lines.append((p2, p4))

    def add_segment(self, p1, p2, color="black"):
        self.segments.append((p1, p2, color))

    def add_line(self, name, line):
        self.lines[name] = line

    def mark_equal_segments(self, p1, p2):
        self.equal_segments.append((p1, p2))

    def _draw_tick_mark(self, ax, p1, p2, num_ticks=1):
        mx, my = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
        dx, dy = p2.x - p1.x, p2.y - p1.y
        length = np.sqrt(dx**2 + dy**2)

        # Vector vuông góc
        nx, ny = -dy / length, dx / length
        tick_size = length * 0.01

        for i in range(num_ticks):
            offset = (i - (num_ticks - 1) / 2) * tick_size * 0.5
            cx = mx + dx / length * offset
            cy = my + dy / length * offset
            ax.plot([cx - nx * tick_size, cx + nx * tick_size],
                   [cy - ny * tick_size, cy + ny * tick_size], 'k-', linewidth=1)

    def plot(self, show=False, save=True, filename="diagram_output.png"):
        fig, ax = plt.subplots(figsize=(8, 8))

        # Vẽ tam giác
        for tri in self.triangles:
            p1, p2, p3 = tri[0], tri[1], tri[2]
            equal_sides = tri[3] if len(tri) > 3 else None

            xs = [p1.x, p2.x, p3.x, p1.x]
            ys = [p1.y, p2.y, p3.y, p1.y]
            ax.plot(xs, ys, 'k-', linewidth=1.5)

            # Vẽ dấu bằng nhau nếu có
            if equal_sides:
                pts = [p1, p2, p3]
                for i, j in equal_sides:
                    self._draw_tick_mark(ax, pts[i], pts[j])

        # Vẽ điểm và tên
        if self.points:
            # Tính centroid của tất cả điểm
            cx = sum(p.x for p in self.points.values()) / len(self.points)
            cy = sum(p.y for p in self.points.values()) / len(self.points)

            for name, p in self.points.items():
                ax.plot(p.x, p.y, 'ko', markersize=4)
                # Đặt label ra ngoài (hướng ngược với centroid)
                dx, dy = p.x - cx, p.y - cy
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    ox, oy = dx / dist * 12, dy / dist * 12
                else:
                    ox, oy = 8, 8
                ax.annotate(name, (p.x, p.y), xytext=(ox, oy),
                           textcoords='offset points', fontsize=10, fontweight='bold')

        # Vẽ đoạn thẳng
        for p1, p2, color in self.segments:
            ax.plot([p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=1.5)

        # Tắt trục tọa độ
        ax.set_aspect('equal')
        ax.axis('off')

        # Auto-scale
        if self.points:
            all_xs = [p.x for p in self.points.values()]
            all_ys = [p.y for p in self.points.values()]
            x_min, x_max = min(all_xs), max(all_xs)
            y_min, y_max = min(all_ys), max(all_ys)
            padding = max(x_max - x_min, y_max - y_min) * 0.15 + 0.1
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)

        plt.tight_layout()

        if save and filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Diagram saved to {filename}")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig, ax