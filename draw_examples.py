"""
GMBL Parser và Optimizer - Mô phỏng geo-model-builder
Parse GMBL và dùng gradient descent để tìm vị trí điểm thỏa mãn constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import re

# VÍ DỤ 1: Tam giác ABC đơn giản
example1 = {
    "instruction": "Tam giác ABC, đường thẳng đi qua điểm B và điểm C",
    "gmbl": """(param (A B C) triangle)
(define line BC (line B C))"""
}

# VÍ DỤ 2: Tam giác với trung điểm
example2 = {
    "instruction": "Tam giác ABC, điểm D là trung điểm của AB, điểm E là trung điểm của AC, điểm F là trung điểm của BC",
    "gmbl": """(param (A B C D E F) triangle)
(define D point (midpoint A B))
(define E point (midpoint A C))
(define F point (midpoint B C))
(define DE line (through D E))
(define EF line (through E F))
(assert (parallel BC DE))
(assert (parallel EF AB))"""
}


class GMBLParser:
    """Parser cho GMBL code"""

    def __init__(self):
        self.points = []  # List point names
        self.constraints = []  # List of constraints
        self.lines = {}  # Line definitions

    def parse(self, gmbl_code: str):
        """Parse GMBL code"""
        lines = gmbl_code.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue

            # Tokenize
            tokens = line.replace('(', ' ( ').replace(')', ' ) ').split()

            if '(' in tokens and len(tokens) > 1:
                cmd = tokens[1]

                if cmd == 'param':
                    self.parse_param(tokens)
                elif cmd == 'define':
                    self.parse_define(tokens)
                elif cmd == 'assert':
                    self.parse_assert(tokens)

    def parse_param(self, tokens: List[str]):
        """Parse param command"""
        # (param (A B C) triangle)
        # Tìm các điểm trong ngoặc
        in_paren = False
        for tok in tokens:
            if tok == '(':
                in_paren = True
            elif tok == ')':
                in_paren = False
            elif in_paren and tok not in ['param']:
                if tok not in self.points:
                    self.points.append(tok)

    def parse_define(self, tokens: List[str]):
        """Parse define command"""
        # (define D point (midpoint A B))
        if len(tokens) < 4:
            return

        name = tokens[2]
        obj_type = tokens[3]

        if name not in self.points:
            self.points.append(name)

        # Tìm constraint
        if 'midpoint' in tokens:
            idx = tokens.index('midpoint')
            p1 = tokens[idx + 1]
            p2 = tokens[idx + 2]
            self.constraints.append({
                'type': 'midpoint',
                'point': name,
                'p1': p1,
                'p2': p2
            })

        if obj_type == 'line':
            # (define BC line (line B C))
            if 'line' in tokens[4:] or 'through' in tokens:
                points_in_line = [t for t in tokens[4:] if t in self.points and t != ')']
                if len(points_in_line) >= 2:
                    self.lines[name] = points_in_line[:2]

    def parse_assert(self, tokens: List[str]):
        """Parse assert command"""
        # (assert (parallel BC DE))
        if 'parallel' in tokens:
            idx = tokens.index('parallel')
            # Lấy 4 điểm tiếp theo
            remaining = tokens[idx+1:]
            points = [t for t in remaining if t in self.points]

            if len(points) >= 4:
                self.constraints.append({
                    'type': 'parallel',
                    'line1': points[:2],
                    'line2': points[2:4]
                })

        elif 'right-angle' in tokens:
            idx = tokens.index('right-angle')
            remaining = tokens[idx+1:]
            points = [t for t in remaining if t in self.points]

            if len(points) >= 3:
                self.constraints.append({
                    'type': 'right-angle',
                    'points': points[:3]
                })


class GeometryOptimizer:
    """Optimizer để tìm vị trí điểm thỏa mãn constraints"""

    def __init__(self, parser: GMBLParser):
        self.parser = parser
        self.point_indices = {name: i for i, name in enumerate(parser.points)}
        self.n_points = len(parser.points)

    def get_point(self, x: np.ndarray, name: str) -> np.ndarray:
        """Lấy tọa độ điểm từ vector x"""
        idx = self.point_indices[name]
        return x[idx*2:idx*2+2]

    def set_point(self, x: np.ndarray, name: str, coords: np.ndarray):
        """Set tọa độ điểm vào vector x"""
        idx = self.point_indices[name]
        x[idx*2:idx*2+2] = coords

    def loss_function(self, x: np.ndarray) -> float:
        """Tính total loss từ tất cả constraints"""
        total_loss = 0.0

        # Loss để giữ tam giác không bị degenerate
        if self.n_points >= 3:
            A = self.get_point(x, self.parser.points[0])
            B = self.get_point(x, self.parser.points[1])
            C = self.get_point(x, self.parser.points[2])

            # Loss: tam giác phải có diện tích > 0
            area = abs((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1]))
            target_area = 10.0  # Target area
            total_loss += (area - target_area) ** 2 * 0.1

            # Loss: các cạnh không quá ngắn hoặc quá dài
            for p1, p2 in [(A, B), (B, C), (C, A)]:
                dist = np.linalg.norm(p1 - p2)
                if dist < 1.0:
                    total_loss += (1.0 - dist) ** 2 * 10
                elif dist > 10.0:
                    total_loss += (dist - 10.0) ** 2 * 0.1

        # Process constraints
        for constraint in self.parser.constraints:
            if constraint['type'] == 'midpoint':
                # D = midpoint(A, B)
                D = self.get_point(x, constraint['point'])
                A = self.get_point(x, constraint['p1'])
                B = self.get_point(x, constraint['p2'])

                midpoint = (A + B) / 2
                loss = np.sum((D - midpoint) ** 2)
                total_loss += loss * 100  # Heavy weight

            elif constraint['type'] == 'parallel':
                # Line1 song song Line2
                p1 = self.get_point(x, constraint['line1'][0])
                p2 = self.get_point(x, constraint['line1'][1])
                p3 = self.get_point(x, constraint['line2'][0])
                p4 = self.get_point(x, constraint['line2'][1])

                # Vector của 2 đường
                v1 = p2 - p1
                v2 = p4 - p3

                # Normalize
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 > 1e-6 and n2 > 1e-6:
                    v1_norm = v1 / n1
                    v2_norm = v2 / n2

                    # Cross product phải = 0 nếu song song
                    cross = abs(v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0])
                    total_loss += cross ** 2 * 50

            elif constraint['type'] == 'right-angle':
                # Góc vuông tại điểm giữa
                p1 = self.get_point(x, constraint['points'][0])
                p2 = self.get_point(x, constraint['points'][1])  # Đỉnh góc vuông
                p3 = self.get_point(x, constraint['points'][2])

                # Vector
                v1 = p1 - p2
                v2 = p3 - p2

                # Dot product phải = 0
                dot = np.dot(v1, v2)
                total_loss += dot ** 2 * 50

        return total_loss

    def optimize(self, n_tries=10) -> np.ndarray:
        """Optimize để tìm vị trí tốt nhất"""
        best_x = None
        best_loss = float('inf')

        for trial in range(n_tries):
            # Random initialization
            x0 = np.random.randn(self.n_points * 2) * 3

            # Optimize
            result = minimize(
                self.loss_function,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000}
            )

            if result.fun < best_loss:
                best_loss = result.fun
                best_x = result.x

        print(f"Optimization: best_loss = {best_loss:.6f}")
        return best_x


def visualize(parser: GMBLParser, x: np.ndarray, title: str, save_path: str):
    """Vẽ kết quả"""
    fig, ax = plt.subplots(figsize=(10, 8))

    optimizer = GeometryOptimizer(parser)

    # Vẽ tam giác chính (3 điểm đầu)
    if len(parser.points) >= 3:
        triangle_points = []
        for i in range(3):
            p = optimizer.get_point(x, parser.points[i])
            triangle_points.append(p)

        triangle = plt.Polygon(triangle_points, fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(triangle)

    # Vẽ các đường thẳng định nghĩa
    for line_name, (p1_name, p2_name) in parser.lines.items():
        p1 = optimizer.get_point(x, p1_name)
        p2 = optimizer.get_point(x, p2_name)

        # Extend line
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length > 0:
            direction = direction / length
            extended_p1 = p1 - direction * 3
            extended_p2 = p2 + direction * 3

            ax.plot([extended_p1[0], extended_p2[0]],
                   [extended_p1[1], extended_p2[1]],
                   'r--', linewidth=1.5, alpha=0.6, label=f'Line {line_name}')

    # Vẽ parallel lines từ constraints
    for constraint in parser.constraints:
        if constraint['type'] == 'parallel':
            # Vẽ line 2
            p1 = optimizer.get_point(x, constraint['line2'][0])
            p2 = optimizer.get_point(x, constraint['line2'][1])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2, alpha=0.7)

    # Vẽ các điểm
    for i, name in enumerate(parser.points):
        p = optimizer.get_point(x, name)

        # 3 điểm đầu màu đỏ, các điểm khác màu xanh
        color = 'red' if i < 3 else 'green'
        size = 10 if i < 3 else 8

        ax.plot(p[0], p[1], 'o', color=color, markersize=size)
        ax.text(p[0] + 0.15, p[1] + 0.15, name,
               fontsize=14 if i < 3 else 12,
               weight='bold', color=color)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11, pad=20)

    # Auto scale
    all_points = [optimizer.get_point(x, name) for name in parser.points]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    margin = 2
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("GMBL PARSER & OPTIMIZER - Tự động tìm vị trí điểm")
    print("="*80 + "\n")

    # VÍ DỤ 1
    print(f"Ví dụ 1: {example1['instruction']}")
    print(f"GMBL:\n{example1['gmbl']}\n")

    parser1 = GMBLParser()
    parser1.parse(example1['gmbl'])
    print(f"Parsed: {len(parser1.points)} points, {len(parser1.constraints)} constraints")

    optimizer1 = GeometryOptimizer(parser1)
    x1 = optimizer1.optimize(n_tries=5)

    visualize(parser1, x1, example1['instruction'], 'example1_optimized.png')

    print("\n" + "-"*80 + "\n")

    # VÍ DỤ 2
    print(f"Ví dụ 2: {example2['instruction']}")
    print(f"GMBL:\n{example2['gmbl']}\n")

    parser2 = GMBLParser()
    parser2.parse(example2['gmbl'])
    print(f"Parsed: {len(parser2.points)} points, {len(parser2.constraints)} constraints")
    print(f"Constraints: {[c['type'] for c in parser2.constraints]}")

    optimizer2 = GeometryOptimizer(parser2)
    x2 = optimizer2.optimize(n_tries=10)

    visualize(parser2, x2, example2['instruction'][:60] + '...', 'example2_optimized.png')

    print("\n" + "="*80)
    print("HOÀN THÀNH - Đã dùng gradient descent tìm vị trí điểm")
    print("="*80)
