# GeoUni - Diagram Builder

> **PyTorch-based Constraint Optimizer** - Chuyển DSL geometry thành hình vẽ tối ưu bằng gradient descent.

## 📁 Diagram Folder Structure

```
llm_src/applications/diagram/
├── optimizer.py           # 875 lines - Core constraint solver
├── initializer.py         # 90 lines - Smart initialization strategies
├── services/
│   ├── dsl_parser.py      # S-expression parser
│   └── diagram_builder.py # DSL → Instructions converter
├── problem.json           # Test problems
└── source.json            # DSL examples
```

## 🔄 Service Flow

```
DSL Text → DSLParser → DiagramBuilder → Optimizer → Rendered Diagram
   │           │              │              │            │
   │           │              │              │            └─> PNG/SVG output
   │           │              │              └─> PyTorch optimization (1000 epochs)
   │           │              └─> Parameter/Assertion instructions
   │           └─> Parse S-expressions → Python tuples
   └─> Input: "(triangle A B C isosceles (at A))"
```

---

## 🔧 Service 1: DSL Parser

**File:** [`services/dsl_parser.py`](llm_src/applications/diagram/services/dsl_parser.py)

### Purpose
Parse S-expression strings → Python tuples

### Class Structure
```python
class DSLParser:
    @classmethod
    def parse_sexprs(cls, lines: List[str]) -> List[Tuple]:
        """Parse multiple DSL lines"""

    def parse_sexpr(self, s: str) -> Tuple:
        """Parse single S-expression"""

    def tokenize(self, s: str) -> List[str]:
        """Split string by ( ) and spaces"""

    def read_from_tokens(self, tokens: List[str]) -> Any:
        """Build nested tuple structure"""
```

### Example
```python
parser = DSLParser()

# Input
dsl = "(triangle A B C isosceles (at A))"

# Step 1: Tokenize
tokens = ['(', 'triangle', 'A', 'B', 'C', 'isosceles', '(', 'at', 'A', ')', ')']

# Step 2: Parse to tuple
result = ('triangle', 'A', 'B', 'C', 'isosceles', ('at', 'A'))
```

### Key Features
- Handles nested structures
- Removes comments (`;` prefix)
- Returns `None` for empty lines

---

## 🔧 Service 2: Diagram Builder

**File:** [`services/diagram_builder.py`](llm_src/applications/diagram/services/diagram_builder.py)

### Purpose
Convert parsed tuples → `Parameter` and `Assertion` instruction objects

### Class Structure
```python
class DiagramBuilder:
    def __init__(self, problem_lines: List[str]):
        self.points: List[Point] = []
        self.instructions: List[Any] = []

    def process_command(self, cmd: Tuple):
        """Route to specific handler"""

    def process_triangle(self, cmd):
        """(triangle A B C isosceles (at A))"""

    def process_define(self, cmd):
        """(define point M (midpoint B C))"""

    def process_segment(self, cmd):
        """(segment A B)"""

    def process_line(self, cmd):
        """(line A B)"""

    def process_circle(self, cmd):
        """(circle I (incircle A B C))"""

    def process_parallel(self, cmd):
        """(parallel (segment B C) (segment D E))"""

    def process_perpendicular(self, cmd):
        """(perpendicular (segment A B) (segment C D))"""
```

### Instruction Types

#### 1. Parameter (Geometry Objects)
```python
# Triangle
Input:  ('triangle', 'A', 'B', 'C', 'isosceles', ('at', 'A'))
Output: Parameter(
    diagram_type=DiagramType.TRIANGLE,
    objects=[Point('A'), Point('B'), Point('C')],
    param_type=TriangleType.ISOSCELES,
    args=(Point('A'),)
)

# Point construction
Input:  ('define', 'M', 'point', ('midpoint', 'B', 'C'))
Output: Parameter(
    diagram_type=DiagramType.POINT,
    objects=[Point('M')],
    param_type='midpoint',
    args=(Point('B'), Point('C'))
)

# Circle
Input:  ('circle', 'I', ('incircle', 'A', 'B', 'C'))
Output: Parameter(
    diagram_type=DiagramType.CIRCLE,
    objects=[Point('I')],
    param_type='incircle',
    args=(Point('A'), Point('B'), Point('C'))
)
```

#### 2. Assertion (Constraints)
```python
# Parallel
Input:  ('parallel', ('segment', 'B', 'C'), ('segment', 'D', 'E'))
Output: Assertion(
    constraint_type='parallel',
    objects=[Point('B'), Point('C'), Point('D'), Point('E')]
)

# Perpendicular
Input:  ('perpendicular', ('segment', 'A', 'B'), ('segment', 'C', 'D'))
Output: Assertion(
    constraint_type='perpendicular',
    objects=[Point('A'), Point('B'), Point('C'), Point('D')]
)
```

### Template Usage
```python
# Parse DSL
dsl_lines = [
    "(triangle A B C isosceles (at A))",
    "(define point M (midpoint B C))",
    "(parallel (segment A M) (segment B C))"
]

# Build instructions
builder = DiagramBuilder(dsl_lines)

# Access results
print(builder.points)        # [Point('A'), Point('B'), Point('C')]
print(builder.instructions)  # [Parameter(...), Parameter(...), Assertion(...)]
```

---

## 🔧 Service 3: Optimizer

**File:** [`optimizer.py`](llm_src/applications/diagram/optimizer.py)

### Purpose
Solve geometric constraints bằng PyTorch gradient descent

### Class Structure
```python
class Optimizer:
    def __init__(self, instructions, opts, verbosity=False):
        self.instructions = instructions  # From DiagramBuilder
        self.opts = opts                  # {epochs, lr, n_tries, eps}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # State
        self.name2pt: Dict[str, TorchPoint] = {}
        self.loss_fns: Dict[str, Callable] = {}
        self.trainable_vars: List[nn.Parameter] = []

    def solve(self, n_tries=1) -> Diagram:
        """Multi-init strategy: try n_tries, return best"""

    def solve_single(self, attempt_id: int) -> Tuple[Diagram, float]:
        """Single optimization attempt"""

    def _init_state(self):
        """Reset state for new attempt"""

    def _process_instructions(self):
        """Build points + constraints from instructions"""

    def mkvar(self, name: str, lo=-1.0, hi=1.0) -> torch.Tensor:
        """Create learnable parameter"""

    def register_loss(self, key: str, loss_fn: Callable, weight: float):
        """Add weighted loss function"""
```

### Key Methods

#### A. Point Creation
```python
def mkvar(self, name, lo=-1.0, hi=1.0, init_value=None):
    """Create learnable nn.Parameter"""
    if init_value is not None:
        val = torch.tensor([init_value], dtype=torch.float64)
    else:
        val = torch.empty(1, dtype=torch.float64).uniform_(lo, hi)
    param = nn.Parameter(val)
    self.trainable_vars.append(param)
    return param.squeeze()

# Usage
x = self.mkvar("A_x", init_value=0.0)
y = self.mkvar("A_y", init_value=1.0)
self.name2pt["A"] = TorchPoint(x, y)
```

#### B. Loss Registration
```python
def register_loss(self, key, loss_fn, weight):
    """Add weighted loss: total_loss += weight * loss_fn()"""
    self.losses[key] = None  # For logging
    self.loss_fns[key] = lambda: weight * (loss_fn() ** 2).mean()

# Usage
self.register_loss(
    "midpoint_M_BC",
    lambda: (self.name2pt["M"].x - (self.name2pt["B"].x + self.name2pt["C"].x) / 2)**2 +
            (self.name2pt["M"].y - (self.name2pt["B"].y + self.name2pt["C"].y) / 2)**2,
    weight=20.0
)
```

#### C. Triangle Processing
```python
def _handle_triangle(self, instr: Parameter):
    """Smart initialization based on triangle type"""
    p1, p2, p3 = instr.objects

    if instr.param_type == TriangleType.ISOSCELES:
        # Symmetric initialization
        coords = Initializer.init_isoceles_triangle()
    elif instr.param_type == TriangleType.RIGHT:
        coords = Initializer.init_right_triangle()
    elif instr.param_type == TriangleType.EQUILATERAL:
        coords = Initializer.init_equilateral_triangle()
    else:
        # Random initialization
        coords = Initializer.init_scalene_triangle()

    # Create learnable points
    for point, (x, y) in zip([p1, p2, p3], coords):
        px = self.mkvar(f"{point.name}_x", init_value=x)
        py = self.mkvar(f"{point.name}_y", init_value=y)
        self.name2pt[point.name] = TorchPoint(px, py)
```

#### D. Special Points
```python
def _handle_special_point(self, instr: Parameter):
    """Create constraint for special points"""
    point_name = instr.objects[0].name
    construction_type = instr.param_type
    args = instr.args

    if construction_type == "midpoint":
        p1, p2 = args
        # Create learnable point
        x = self.mkvar(f"{point_name}_x")
        y = self.mkvar(f"{point_name}_y")
        self.name2pt[point_name] = TorchPoint(x, y)

        # Add midpoint constraint
        def midpoint_loss():
            m = self.name2pt[point_name]
            a = self.name2pt[p1.name]
            b = self.name2pt[p2.name]
            return ((m.x - (a.x + b.x) / 2) ** 2 +
                    (m.y - (a.y + b.y) / 2) ** 2)

        self.register_loss(f"midpoint_{point_name}", midpoint_loss, weight=20.0)

    elif construction_type == "incenter":
        # Distance from incenter to 3 sides equal
        # ... (similar pattern)
```

#### E. Constraints
```python
def _handle_constraint(self, assertion: Assertion):
    """Process parallel/perpendicular constraints"""
    constraint_type = assertion.constraint_type
    points = assertion.objects

    if constraint_type == "parallel":
        p1, p2, p3, p4 = points  # BC parallel DE

        def parallel_loss():
            # Vector BC
            dx1 = self.name2pt[p2.name].x - self.name2pt[p1.name].x
            dy1 = self.name2pt[p2.name].y - self.name2pt[p1.name].y

            # Vector DE
            dx2 = self.name2pt[p4.name].x - self.name2pt[p3.name].x
            dy2 = self.name2pt[p4.name].y - self.name2pt[p3.name].y

            # Cross product = 0
            cross = dx1 * dy2 - dy1 * dx2
            return cross ** 2

        self.register_loss(f"parallel_{p1.name}{p2.name}_{p3.name}{p4.name}",
                          parallel_loss, weight=50.0)
```

### Optimization Loop
```python
def solve_single(self, attempt_id):
    """Single optimization run"""
    self.current_attempt = attempt_id

    # Setup optimizer
    optimizer = optim.Adam(self.trainable_vars, lr=self.opts.get('lr', 0.01))

    # Training loop
    for epoch in range(self.opts.get('epochs', 1000)):
        optimizer.zero_grad()

        # Compute total loss
        total_loss = sum(fn() for fn in self.loss_fns.values())

        # Backward + step
        total_loss.backward()
        optimizer.step()

        # Early stopping
        if total_loss.item() < self.opts.get('eps', 1e-8):
            break

    # Build diagram from optimized points
    diagram = self._build_diagram()
    return diagram, total_loss.item()
```

### Loss Weights (Current)
```python
WEIGHTS = {
    'regularization': 0.001,      # Keep points near origin
    'parallel': 50.0,             # Parallel constraint
    'perpendicular': 10.0,        # Perpendicular constraint
    'midpoint': 20.0,             # Midpoint constraint
    'incenter': 50.0,             # Incenter distances equal
    'circumcenter': 50.0,         # Circumcenter distances equal
    'collinear': 10.0,            # Points on line
    'equal_distance': 10.0,       # Two distances equal
}
```

---

## 📝 DSL Syntax Reference

### Triangles
```scheme
(triangle A B C)                         # Scalene
(triangle A B C isosceles (at A))        # Isosceles at A
(triangle A B C right (at B))            # Right at B
(triangle A B C equilateral)             # Equilateral
(triangle A B C right-isosceles (at C))  # Right isosceles at C
```

### Points
```scheme
(define point M (midpoint B C))          # Midpoint
(define point I (incenter A B C))        # Incenter
(define point O (circumcenter A B C))    # Circumcenter
(define point H (orthocenter A B C))     # Orthocenter
(define point G (centroid A B C))        # Centroid
(define point P (on-segment A B))        # On segment (learnable parameter)
```

### Geometry Objects
```scheme
(segment A B)                            # Draw segment
(line A B)                               # Draw line (infinite)
(circle I (incircle A B C))             # Incircle
(circle O (circumcircle A B C))         # Circumcircle
```

### Constraints
```scheme
(parallel (segment B C) (segment D E))   # BC ∥ DE
(perpendicular (segment A B) (segment C D))  # AB ⊥ CD
```

---

## 🚀 Usage Template

```python
from llm_src.applications.diagram.optimizer import Optimizer

# Step 1: Define DSL
dsl_lines = [
    "(triangle A B C isosceles (at A))",
    "(define point M (midpoint B C))",
    "(segment A M)",
]

# Step 2: Configure optimizer
opts = {
    'n_tries': 3,           # Multi-init attempts
    'epochs': 1000,         # Max epochs per attempt
    'learning_rate': 0.01,  # Adam learning rate
    'eps': 1e-8,            # Early stopping threshold
}

# Step 3: Optimize
optimizer = Optimizer(dsl_lines, opts, verbosity=False)
diagram = optimizer.solve(n_tries=opts['n_tries'])

# Step 4: Render
diagram.render("output/result.png")

# Step 5: Check loss
print(f"Final loss: {optimizer.losses}")
# Output: {'regularization': 0.0001, 'midpoint_M': 0.0001, 'isosceles_AB_AC': 0.0002}
```

### Advanced: Access Optimized Points
```python
# Get final coordinates
for name, pt in optimizer.name2pt.items():
    print(f"{name}: ({pt.x.item():.3f}, {pt.y.item():.3f})")

# Output:
# A: (0.000, 1.732)
# B: (-1.000, 0.000)
# C: (1.000, 0.000)
# M: (0.000, 0.000)
```

