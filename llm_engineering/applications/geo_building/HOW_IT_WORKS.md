# HƯỚNG DẪN CHI TIẾT: Cách test và luồng chạy geo_building

## 📋 MỤC LỤC
1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Giải thích từng file](#2-giải-thích-từng-file)
3. [Luồng chạy chi tiết](#3-luồng-chạy-chi-tiết)
4. [Cách test](#4-cách-test)
5. [Ví dụ debug từng bước](#5-ví-dụ-debug-từng-bước)

---

## 1. TỔNG QUAN KIẾN TRÚC

```
Input (GMBL file)
    ↓
[parse.py] → Parse S-expressions
    ↓
[instruction_reader.py] → Đọc và convert thành instructions
    ↓
[builder.py] → Thực thi instructions
    ↓ (sử dụng)
[primitives.py] → Point, Line objects
[constraint.py] → Geometric constraints
[util.py] → Helper functions
    ↓
[scipy.optimize] → Tối ưu hóa vị trí điểm
    ↓
[diagram.py] → Vẽ kết quả
    ↓
Output (Matplotlib plot)
```

---

## 2. GIẢI THÍCH TỪNG FILE

### 📄 `primitives.py` - Các đối tượng hình học cơ bản
**Chức năng**: Định nghĩa Point, Line, Num

```python
Point("A")        # Điểm có tên A
Line("l")         # Đường thẳng có tên l
Num(5.0)          # Số thực 5.0
```

**Khi nào chạy**: Được import bởi tất cả các file khác, chạy đầu tiên khi import

---

### 📄 `parse.py` - Parser S-expressions
**Chức năng**: Chuyển đổi text GMBL → Python tuples

**Input**:
```
(param (A B C) equi-tri)
```

**Output**:
```python
('param', ('A', 'B', 'C'), 'equi-tri')
```

**Luồng xử lý**:
```
1. tokenize("(param (A B C) equi-tri)")
   → ['(', 'param', '(', 'A', 'B', 'C', ')', 'equi-tri', ')']

2. read_from_tokens(tokens)
   → ('param', ('A', 'B', 'C'), 'equi-tri')
```

**Khi nào chạy**: Bước đầu tiên trong InstructionReader.__init__()

---

### 📄 `util.py` - Hàm tiện ích
**Chức năng**:
- `FuncInfo`: Lưu thông tin hàm (head, args)
- `Root`: Đại diện cho root constraint
- `is_number()`: Kiểm tra string có phải số không
- `DEFAULTS`: Các tham số mặc định

**Khi nào chạy**: Import vào các file khác, sử dụng xuyên suốt

---

### 📄 `constraint.py` - Ràng buộc hình học
**Chức năng**: Đại diện cho các constraint như:
- `cong`: Hai điểm trùng nhau
- `perp`: Hai đường thẳng vuông góc
- `para`: Hai đường thẳng song song
- `midp`: Điểm là trung điểm
- `coll`: Ba điểm thẳng hàng

**Khi nào chạy**:
1. Được tạo trong InstructionReader khi parse assertions
2. Được sử dụng trong builder.py để tối ưu hóa

---

### 📄 `instruction.py` - Các loại instruction
**Chức năng**: Định nghĩa 5 loại instruction:

```python
# 1. Sample - Khởi tạo tam giác ngẫu nhiên
Sample(
    points=[Point('A'), Point('B'), Point('C')],
    sampler='equi-tri',
    args=()
)

# 2. Parameterize - Khởi tạo điểm/đường thẳng từ tọa độ
Parameterize(
    obj_name=Point('D'),
    parameterization=FuncInfo('coords', [])
)

# 3. Compute - Tính toán đối tượng mới
Compute(
    obj_name=Point('M'),
    computation=FuncInfo('midp', [Point('B'), Point('C')])
)

# 4. Assert - Khẳng định constraint
Assert(
    constraint=Constraint('perp', [...])
)

# 5. Eval - Đánh giá constraint (goal)
Eval(
    constraint=Constraint('cong', [...])
)
```

**Khi nào chạy**: Được tạo bởi InstructionReader, thực thi bởi Builder

---

### 📄 `instruction_reader.py` - Đọc và parse instructions
**Chức năng**: Chuyển S-expressions → Instruction objects

**Luồng xử lý**:

```python
Input: "(param (A B C) equi-tri)"

1. parse_sexprs()
   → [('param', ('A', 'B', 'C'), 'equi-tri')]

2. process_command()
   → Nhận diện head='param'
   → Gọi process_param_special()

3. process_param_special()
   → Tạo Point('A'), Point('B'), Point('C')
   → register_pt() cho từng điểm
   → Tạo Sample(points=[A,B,C], sampler='equi-tri')
   → Thêm segments [(A,B), (B,C), (C,A)]

4. Kết quả:
   self.points = [Point('A'), Point('B'), Point('C')]
   self.instructions = [Sample(...)]
   self.segments = [(A,B), (B,C), (C,A)]
```

**Các phương thức chính**:
- `process_command()`: Router đến các handler
- `process_param_special()`: Xử lý sampling tam giác
- `compute()`: Xử lý define point/line
- `add()`: Xử lý assert
- `process_point()`: Parse point computation
- `process_line()`: Parse line computation
- `process_constraint()`: Parse constraint

**Khi nào chạy**: Ngay khi khởi tạo InstructionReader trong Builder.__init__()

---

### 📄 `builder.py` - Core logic: thực thi và tối ưu
**Chức năng**:
1. Thực thi instructions
2. Tối ưu hóa vị trí điểm theo constraints
3. Tạo Diagram object

**Luồng xử lý**:

```python
class SimpleBuilder:
    def __init__(self, lines, optimize=True):
        # Bước 1: Parse instructions
        self.reader = InstructionReader(lines)

    def build(self):
        # Bước 2: Thực thi instructions (pass 1)
        for instr in self.reader.instructions:
            if isinstance(instr, Sample):
                self.sample_triangle(...)  # Khởi tạo tam giác
            elif isinstance(instr, Parameterize):
                self.parameterize(...)     # Khởi tạo điểm coords
            elif isinstance(instr, Compute):
                self.compute(...)          # Tính điểm mới

        # Bước 3: Thu thập constraints (pass 2)
        for instr in self.reader.instructions:
            if isinstance(instr, Assert):
                self.constraints.append(instr.constraint)

        # Bước 4: Tối ưu hóa
        if self.optimize and self.constraints:
            self.optimize_points()  # Dùng scipy

        # Bước 5: Tạo Diagram
        return Diagram(...)
```

**Chi tiết các phương thức**:

#### `sample_triangle()` - Khởi tạo tam giác
```python
# Ví dụ: equi-tri
def sample_triangle(self, points, sampler, args):
    if sampler == 'equi-tri':
        # Tạo tam giác đều
        A = SimplePoint(0, 0)
        B = SimplePoint(1, 0)
        C = SimplePoint(0.5, 0.866)  # sqrt(3)/2

        self.named_points[Point('A')] = A
        self.named_points[Point('B')] = B
        self.named_points[Point('C')] = C
```

#### `compute()` - Tính điểm mới
```python
# Ví dụ: midp
def compute(self, instr):
    if computation.head == 'midp':
        p1 = self.lookup_point(args[0])  # Point B
        p2 = self.lookup_point(args[1])  # Point C

        # Trung điểm
        mid = SimplePoint(
            (p1.x + p2.x) / 2,
            (p1.y + p2.y) / 2
        )

        self.named_points[Point('M')] = mid
```

#### `optimize_points()` - Tối ưu hóa
```python
def optimize_points(self):
    # 1. Chuyển points thành vector
    x0 = [pt.x, pt.y for pt in all_points]  # [x1,y1,x2,y2,...]

    # 2. Định nghĩa loss function
    def loss(x):
        loss_total = 0
        for constraint in self.constraints:
            # Tính lỗi cho từng constraint
            loss_total += constraint_loss(constraint, x)
        return loss_total

    # 3. Tối ưu hóa bằng scipy
    result = minimize(
        loss,
        x0,
        method='BFGS',
        options={'maxiter': 100}
    )

    # 4. Cập nhật lại points
    optimized_x = result.x
    for i, point in enumerate(all_points):
        point.x = optimized_x[2*i]
        point.y = optimized_x[2*i + 1]
```

**Khi nào chạy**: Khi gọi build_from_file() hoặc build()

---

### 📄 `diagram.py` - Vẽ kết quả
**Chức năng**: Sử dụng matplotlib để vẽ:
- Điểm (points)
- Đường thẳng (lines)
- Đoạn thẳng (segments)
- Labels

```python
class Diagram:
    def plot(self, show=True, save=False, fname='diagram.png'):
        # Vẽ segments
        for (p1, p2), color in zip(self.segments, self.seg_colors):
            plt.plot([p1.x, p2.x], [p1.y, p2.y], color=color)

        # Vẽ points
        for point, coords in self.named_points.items():
            plt.scatter(coords.x, coords.y)
            plt.text(coords.x, coords.y, point.val)

        plt.show() if show else plt.savefig(fname)
```

**Khi nào chạy**: Cuối cùng sau khi build() hoàn thành

---

### 📄 `main.py` - Entry point CLI
**Chức năng**: Command line interface

```bash
python main.py test_triangle.gmbl
python main.py test_triangle.gmbl --save output.png
python main.py test_triangle.gmbl --no-plot
```

**Khi nào chạy**: Khi user chạy từ terminal

---

### 📄 `example.py` - Ví dụ sử dụng
**Chức năng**: Các ví dụ hardcoded để test

```bash
python example.py
```

**Khi nào chạy**: Để test nhanh các tính năng

---

## 3. LUỒNG CHẠY CHI TIẾT

### Ví dụ: File `test_triangle.gmbl`
```
(param (A B C) equi-tri)
```

### Bước 1: Parsing (parse.py)
```python
Input: "(param (A B C) equi-tri)"

tokenize()
→ ['(', 'param', '(', 'A', 'B', 'C', ')', 'equi-tri', ')']

read_from_tokens()
→ ('param', ('A', 'B', 'C'), 'equi-tri')
```

### Bước 2: Instruction Reading (instruction_reader.py)
```python
__init__():
    cmds = parse_sexprs(lines)
    # cmds = [('param', ('A', 'B', 'C'), 'equi-tri')]

    for cmd in cmds:
        process_command(cmd)

process_command(cmd):
    head = 'param'
    cmd[1] = ('A', 'B', 'C')  # tuple → process_param_special

process_param_special(cmd):
    ps = [Point('A'), Point('B'), Point('C')]

    for p in ps:
        register_pt(p)  # Thêm vào self.points

    instr = Sample(ps, 'equi-tri', ())
    self.instructions.append(instr)

    # Thêm segments
    self.segments = [
        (Point('A'), Point('B')),
        (Point('B'), Point('C')),
        (Point('C'), Point('A'))
    ]
```

**Kết quả sau bước 2**:
```python
reader.points = [Point('A'), Point('B'), Point('C')]
reader.instructions = [
    Sample(
        points=[Point('A'), Point('B'), Point('C')],
        sampler='equi-tri',
        args=()
    )
]
reader.segments = [(A,B), (B,C), (C,A)]
```

### Bước 3: Building (builder.py)
```python
SimpleBuilder.__init__(lines):
    self.reader = InstructionReader(lines)  # Bước 2 xảy ra ở đây
    self.named_points = {}
    self.constraints = []

SimpleBuilder.build():
    # Pass 1: Thực thi instructions
    for instr in self.reader.instructions:
        # instr = Sample(points=[A,B,C], sampler='equi-tri')
        sample_triangle(instr.points, instr.sampler, instr.args)

sample_triangle([A,B,C], 'equi-tri', ()):
    # Tạo tam giác đều
    p0 = SimplePoint(0.0, 0.0)
    p1 = SimplePoint(1.0, 0.0)
    p2 = SimplePoint(0.5, 0.866)

    self.named_points[Point('A')] = p0
    self.named_points[Point('B')] = p1
    self.named_points[Point('C')] = p2
```

**Kết quả sau bước 3**:
```python
builder.named_points = {
    Point('A'): SimplePoint(0.0, 0.0),
    Point('B'): SimplePoint(1.0, 0.0),
    Point('C'): SimplePoint(0.5, 0.866)
}
```

### Bước 4: Optimization (nếu có constraints)
```python
# Trong ví dụ này không có constraints nên bỏ qua
```

### Bước 5: Diagram Creation
```python
return Diagram(
    named_points=self.named_points,  # {A: (0,0), B: (1,0), C: (0.5,0.866)}
    segments=[(A,B), (B,C), (C,A)],
    ...
)
```

### Bước 6: Plotting (diagram.py)
```python
diagram.plot(show=True):
    # Vẽ 3 đoạn thẳng AB, BC, CA
    # Vẽ 3 điểm A, B, C
    # Hiển thị labels
    plt.show()
```

---

## 4. CÁCH TEST

### Test 1: Chạy example đơn giản nhất
```bash
cd d:\projects\GeoUni\llm_engineering\applications\geo_building
python example.py
```

**Kỳ vọng**: Hiển thị 5-7 cửa sổ matplotlib với các tam giác

### Test 2: Chạy file GMBL
```bash
python main.py test_triangle.gmbl
```

**Kỳ vọng**: Hiển thị tam giác đều ABC

### Test 3: Test từng component riêng lẻ

#### Test parse.py
```python
from parse import parse_sexprs

lines = ["(param (A B C) equi-tri)"]
result = parse_sexprs(lines)
print(result)
# Kỳ vọng: [('param', ('A', 'B', 'C'), 'equi-tri')]
```

#### Test instruction_reader.py
```python
from instruction_reader import InstructionReader

lines = ["(param (A B C) equi-tri)"]
reader = InstructionReader(lines)
print(f"Points: {reader.points}")
print(f"Instructions: {reader.instructions}")
print(f"Segments: {reader.segments}")
```

#### Test builder.py
```python
from builder import build

lines = ["(param (A B C) equi-tri)"]
diagram = build(lines, show_plot=False)
print(f"Named points: {len(diagram.named_points)}")
print(f"Segments: {len(diagram.segments)}")
```

### Test 4: Test với constraint phức tạp
```python
# Tạo file test_midpoint.gmbl
lines = [
    "(param (A B C) triangle)",
    "(define M point (midp B C))"
]

from builder import build
diagram = build(lines, show_plot=True)
```

**Kỳ vọng**:
- Tam giác ABC ngẫu nhiên
- Điểm M nằm ở giữa BC

### Test 5: Test optimization
```python
lines = [
    "(param (A B C) triangle)",
    "(define M point (midp B C))",
    "(assert (perp (connect A M) (connect B C)))"  # AM vuông góc BC
]

diagram = build(lines, show_plot=True)
# Sau optimization, AM phải vuông góc BC
```

---

## 5. VÍ DỤ DEBUG TỪNG BƯỚC

### Scenario: File phức tạp hơn
```
(param (A B C) equi-tri)
(define M point (midp B C))
(define l line (mediator B C))
```

### Debug bằng print statements:

#### Thêm vào instruction_reader.py
```python
def process_command(self, cmd):
    print(f"[DEBUG] Processing command: {cmd}")
    head = cmd[0].lower()
    # ... rest of code

def process_param_special(self, cmd):
    print(f"[DEBUG] Sampling triangle with: {cmd[2]}")
    # ... rest of code

def compute(self, cmd):
    print(f"[DEBUG] Computing {cmd[1]} of type {cmd[2]}")
    # ... rest of code
```

#### Thêm vào builder.py
```python
def sample_triangle(self, points, sampler, args):
    print(f"[DEBUG] Creating triangle: {[p.val for p in points]}, type={sampler}")
    # ... rest of code

def compute(self, instr):
    print(f"[DEBUG] Computing {instr.obj_name.val}")
    computation = instr.computation
    print(f"[DEBUG]   Function: {computation.head}, Args: {computation.args}")
    # ... rest of code
```

### Chạy với debug:
```bash
python main.py test_centers.gmbl
```

**Output kỳ vọng**:
```
[DEBUG] Processing command: ('param', ('A', 'B', 'C'), 'equi-tri')
[DEBUG] Sampling triangle with: equi-tri
[DEBUG] Creating triangle: ['A', 'B', 'C'], type=equi-tri
[DEBUG] Processing command: ('define', 'M', 'point', ('midp', 'B', 'C'))
[DEBUG] Computing M of type point
[DEBUG] Computing M
[DEBUG]   Function: midp, Args: [Point('B'), Point('C')]
[DEBUG] Processing command: ('define', 'l', 'line', ('mediator', 'B', 'C'))
[DEBUG] Computing l of type line
Built diagram with 4 named points and 1 named lines
```

---

## 6. KIẾN TRÚC LUỒNG DỮ LIỆU

```
GMBL Text
    ↓
┌─────────────────────────────────────┐
│ parse.py: tokenize + read_tokens   │
└─────────────────────────────────────┘
    ↓ Python tuples
┌─────────────────────────────────────┐
│ instruction_reader.py               │
│ ┌─────────────────────────────────┐ │
│ │ process_param_special()         │ │ → Sample instructions
│ │ compute()                       │ │ → Compute instructions
│ │ add()                           │ │ → Assert instructions
│ │ eval_cons()                     │ │ → Eval instructions
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓ Instruction objects
┌─────────────────────────────────────┐
│ builder.py                          │
│ ┌─────────────────────────────────┐ │
│ │ Pass 1: Execute instructions    │ │
│ │   sample_triangle()             │ │
│ │   parameterize()                │ │
│ │   compute()                     │ │
│ │                                 │ │
│ │ Pass 2: Collect constraints     │ │
│ │                                 │ │
│ │ Pass 3: Optimize (scipy)        │ │
│ │   optimize_points()             │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
    ↓ Diagram object
┌─────────────────────────────────────┐
│ diagram.py: plot()                  │
└─────────────────────────────────────┘
    ↓ matplotlib figure
   USER
```

---

## TÓM TẮT: Để viết lại, bạn cần hiểu

### 1. **Data flow**: GMBL text → tuples → Instructions → Points/Lines → Diagram
### 2. **Key abstractions**:
   - `Point`, `Line`: Symbolic names
   - `SimplePoint`: Concrete (x, y) coordinates
   - `Instruction`: Actions to execute
   - `Constraint`: Geometric relationships to satisfy
### 3. **Two-phase execution**:
   - Phase 1: Initialize geometry from instructions
   - Phase 2: Optimize to satisfy constraints
### 4. **Separation of concerns**:
   - Parsing (parse.py)
   - Semantic analysis (instruction_reader.py)
   - Execution (builder.py)
   - Visualization (diagram.py)

---

## NEXT STEPS

1. **Chạy test đơn giản**: `python example.py`
2. **Thêm debug prints**: Như section 5
3. **Test từng component**: Như section 4.3
4. **Đọc kỹ builder.py**: File quan trọng nhất
5. **Thử modify**: Thêm shape mới hoặc constraint mới
