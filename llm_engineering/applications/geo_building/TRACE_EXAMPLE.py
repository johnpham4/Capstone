from asymptote_builder import AsymptoteBuilder

code = "(triangle A B C (isosceles A))"
lines = [code]

print("=" * 60)
print("BƯỚC 1: Khởi tạo AsymptoteBuilder")
print("=" * 60)
print(f"Input lines: {lines}")
print()

builder = AsymptoteBuilder(lines, optimize=False)
# → Gọi AsymptoteParser(lines)
# → parser.parse_all()

print("Sau khi parse:")
print(f"  parser.points = {[str(p) for p in builder.parser.points]}")
print(f"  parser.instructions = {builder.parser.instructions}")
print(f"  parser.segments = {[(str(p1), str(p2)) for p1, p2 in builder.parser.segments]}")
print()

# ============================================================
# BƯỚC 2: BUILD DIAGRAM
# ============================================================
print("=" * 60)
print("BƯỚC 2: Build diagram")
print("=" * 60)

# builder.build() sẽ chạy:

# Khởi tạo biến
print("Khởi tạo biến tracking:")
builder.computed_points = set()
builder.on_segment_constraints = {}
print(f"  computed_points = {builder.computed_points}")
print(f"  on_segment_constraints = {builder.on_segment_constraints}")
print()

# Xử lý instruction: Sample(points=[A, B, C], sampler='iso-tri', args=(A,))
instr = builder.parser.instructions[0]
print(f"Xử lý instruction: {instr}")
print(f"  Type: Sample")
print(f"  points: {[str(p) for p in instr.points]}")
print(f"  sampler: {instr.sampler}")
print(f"  args: {instr.args}")
print()

# Gọi sample_triangle()
print("Gọi sample_triangle()")
print("  Code chạy:")
print("    sampler == 'iso-tri'")
print("    special_p = Point('A')  # Đỉnh cân")
print("    special_idx = 0")
print("    remaining = [Point('B'), Point('C')]  # Đáy")
print()
print("  Khởi tạo tọa độ:")
print("    base_y = -0.5  (random từ -1 đến 0)")
print("    B.x = -1.5, B.y = -0.5  (đáy trái)")
print("    C.x = 1.8, C.y = -0.5   (đáy phải)")
print("    base_mid_x = (-1.5 + 1.8) / 2 = 0.15")
print("    A.x = 0.15, A.y = -0.5 + 2.5 = 2.0  (đỉnh trên đường cao)")
print()

# Thực tế chạy (giả lập)
import numpy as np
np.random.seed(42)  # Fix seed để kết quả giống nhau

from primitives import Point
from asymptote_builder import SimplePoint

points = [Point('A'), Point('B'), Point('C')]
sampler = 'iso-tri'
args = (Point('A'),)

special_p = args[0]
special_idx = points.index(special_p)  # 0

remaining = [p for i, p in enumerate(points) if i != special_idx]  # [B, C]
base_y = np.random.uniform(-1, 0)  # -0.25

builder.named_points = {}
builder.named_points[remaining[0]] = SimplePoint(np.random.uniform(-2, -1), base_y)  # B
builder.named_points[remaining[1]] = SimplePoint(np.random.uniform(1, 2), base_y)    # C

base_mid_x = (builder.named_points[remaining[0]].x + builder.named_points[remaining[1]].x) / 2
builder.named_points[points[special_idx]] = SimplePoint(
    base_mid_x,
    base_y + np.random.uniform(2, 3)
)

print("KẾT QUẢ named_points:")
for p, sp in builder.named_points.items():
    print(f"  {p}: ({sp.x:.3f}, {sp.y:.3f})")
print()

# ============================================================
# BƯỚC 3: KIỂM TRA CÁC BIẾN
# ============================================================
print("=" * 60)
print("BƯỚC 3: Kiểm tra các biến")
print("=" * 60)

print(f"named_points (điểm có tên):")
for point, coord in builder.named_points.items():
    print(f"  {point.val}: SimplePoint(x={coord.x:.3f}, y={coord.y:.3f})")
print()

print(f"unnamed_points (điểm không tên): {builder.unnamed_points}")
print(f"  → List rỗng vì KHÔNG CÓ điểm phụ trợ nào được tạo")
print()

print(f"named_lines (đường thẳng có tên): {builder.named_lines}")
print(f"  → Dictionary rỗng vì không khai báo line nào")
print()

print(f"unnamed_lines (đường phụ): {builder.unnamed_lines}")
print(f"  → List rỗng vì không có đường phụ nào")
print()

print(f"segments (các đoạn thẳng vẽ):")
for (p1, p2), color in zip(builder.parser.segments, builder.parser.seg_colors):
    print(f"  ({p1.val}, {p2.val}) - color: {color}")
print(f"  → 3 cạnh tam giác: AB, BC, CA")
print()

print(f"computed_points (điểm tính toán): {builder.computed_points}")
print(f"  → Set rỗng vì A, B, C đều là FREE POINTS (không computed)")
print()

# ============================================================
# BƯỚC 4: TẠO DIAGRAM
# ============================================================
print("=" * 60)
print("BƯỚC 4: Tạo Diagram object")
print("=" * 60)

segments = []
for p1, p2 in builder.parser.segments:
    sp1 = builder.named_points.get(p1)
    sp2 = builder.named_points.get(p2)
    if sp1 and sp2:
        segments.append((sp1, sp2))

print(f"segments (converted to SimplePoint tuples):")
for seg in segments:
    print(f"  ({seg[0].x:.2f}, {seg[0].y:.2f}) → ({seg[1].x:.2f}, {seg[1].y:.2f})")
print()

from diagram import Diagram

diagram = Diagram(
    named_points=builder.named_points,      # {Point('A'): SimplePoint(...), ...}
    named_lines=builder.named_lines,        # {}
    segments=segments,                       # [(SimplePoint, SimplePoint), ...]
    seg_colors=builder.parser.seg_colors,   # [array([0.3, 0.7, 0.2]), ...]
    unnamed_points=builder.unnamed_points,  # []
    unnamed_lines=builder.unnamed_lines,    # []
    ndgs={},
    goals={}
)

print("Diagram object created:")
print(f"  named_points: {len(diagram.named_points)} điểm")
print(f"  unnamed_points: {len(diagram.unnamed_points)} điểm")
print(f"  named_lines: {len(diagram.named_lines)} đường")
print(f"  unnamed_lines: {len(diagram.unnamed_lines)} đường")
print(f"  segments: {len(diagram.segments)} đoạn thẳng")
print()

# ============================================================
# BƯỚC 5: VẼ HÌNH
# ============================================================
print("=" * 60)
print("BƯỚC 5: Vẽ hình (diagram.plot)")
print("=" * 60)

print("Code trong diagram.plot():")
print("""
    # Lấy tọa độ các điểm CÓ TÊN
    xs = [p.x for p in self.named_points.values()]  # [0.15, -1.5, 1.8]
    ys = [p.y for p in self.named_points.values()]  # [2.0, -0.25, -0.25]
    names = [n for n in self.named_points.keys()]   # [Point('A'), Point('B'), Point('C')]

    # Vẽ điểm
    ax.scatter(xs, ys, s=30, zorder=5)

    # Gán nhãn
    for i, n in enumerate(names):
        ax.annotate('A', (0.15, 2.0), xytext=(5, 5), ...)
        ax.annotate('B', (-1.5, -0.25), xytext=(5, 5), ...)
        ax.annotate('C', (1.8, -0.25), xytext=(5, 5), ...)

    # Vẽ unnamed_points (KHÔNG CÓ trong ví dụ này)
    if unnamed_points:  # []
        ax.scatter(..., alpha=0.1)  # Mờ hơn

    # Vẽ các cạnh tam giác
    for (p1, p2), c in zip(segments, seg_colors):
        plt.plot([p1.x, p2.x], [p1.y, p2.y], c=c, linewidth=2)
        # AB: (-1.5, -0.25) → (0.15, 2.0)
        # BC: (1.8, -0.25) → (0.15, 2.0)
        # CA: (0.15, 2.0) → (-1.5, -0.25)

    # Tắt axis
    plt.axis('off')

    # Lưu file
    plt.savefig('triangle.png')
""")

print()
print("=" * 60)
print("KẾT QUẢ CUỐI CÙNG")
print("=" * 60)
print("Hình vẽ sẽ có:")
print("  - 3 điểm A, B, C với NHÃN rõ ràng")
print("  - 3 cạnh AB, BC, CA nối các điểm")
print("  - KHÔNG CÓ điểm/đường unnamed (mờ)")
print("  - Tam giác CÂN tại A (AB = AC)")
