import torch
from collections import namedtuple

TorchPoint = namedtuple("TorchPoint", ["x", "y"])
LineSF = namedtuple("LineSF", ["a", "b", "c", "p1", "p2"])
LineNF = namedtuple("LineNF", ["n", "f"])


def dist(p1: TorchPoint, p2: TorchPoint) -> torch.Tensor:
    """Euclidean distance between two points."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return torch.sqrt(dx ** 2 + dy ** 2)


def norm(p: TorchPoint) -> torch.Tensor:
    """Distance from point to origin."""
    return torch.sqrt(p.x ** 2 + p.y ** 2)


def vector_from_points(pa: TorchPoint, pb: TorchPoint):
    """Vector from *pb* to *pa*: (pa − pb)."""
    return (pa.x - pb.x, pa.y - pb.y)


def vec_length(vx, vy) -> torch.Tensor:
    """Length of a 2-D vector given its components."""
    return torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)


def segment_length(p1: TorchPoint, p2: TorchPoint) -> torch.Tensor:
    """Length of segment from *p1* to *p2* (with numerical guard)."""
    vx, vy = vector_from_points(p2, p1)
    return torch.sqrt(vx ** 2 + vy ** 2 + 1e-8)


def pp2lnf(p1: TorchPoint, p2: TorchPoint) -> LineNF:
    """Two-point → normal-form line (n · p = f)."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y

    n_x = -dy
    n_y = dx

    n_norm = torch.sqrt(n_x ** 2 + n_y ** 2)
    n_x = n_x / n_norm
    n_y = n_y / n_norm

    # Ensure normal points to upper half-plane
    if n_y < 0:
        n_x = -n_x
        n_y = -n_y

    f = n_x * p1.x + n_y * p1.y
    n = TorchPoint(n_x, n_y)
    return LineNF(n, f)


def on_line(p: TorchPoint, line: LineNF) -> torch.Tensor:
    """Signed distance of *p* from *line* (0 when on the line)."""
    return line.n.x * p.x + line.n.y * p.y - line.f

def perpendicular(p1: TorchPoint, p2: TorchPoint,
                  p3: TorchPoint, p4: TorchPoint) -> torch.Tensor:
    """(p1−p2) ⊥ (p3−p4). Returns 0 when perpendicular."""
    return (p1.x - p2.x) * (p3.x - p4.x) + (p1.y - p2.y) * (p3.y - p4.y)


def parallel(p1: TorchPoint, p2: TorchPoint,
             p3: TorchPoint, p4: TorchPoint) -> torch.Tensor:
    """(p1−p2) ∥ (p3−p4). Returns 0 when parallel."""
    return (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x)


def collinear(p1: TorchPoint, p2: TorchPoint,
              p3: TorchPoint) -> torch.Tensor:
    """Returns 0 when p1, p2, p3 are collinear."""
    return p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)


def dot_product(pa: TorchPoint, pb: TorchPoint,
                pc: TorchPoint) -> torch.Tensor:
    """Dot product of vectors (pa−pb) · (pc−pb)."""
    v1x, v1y = vector_from_points(pa, pb)
    v2x, v2y = vector_from_points(pc, pb)
    return v1x * v2x + v1y * v2y


def cross_product_area(p1: TorchPoint, p2: TorchPoint,
                       p3: TorchPoint) -> torch.Tensor:
    """Cross product for area / collinearity check."""
    v1x, v1y = vector_from_points(p2, p1)
    v2x, v2y = vector_from_points(p3, p1)
    return v1x * v2y - v1y * v2x


def angle_cosine(p1: TorchPoint, vertex: TorchPoint,
                 p2: TorchPoint) -> torch.Tensor:
    """Cosine of angle ∠p1-vertex-p2."""
    dot = dot_product(p1, vertex, p2)
    v1_x, v1_y = vector_from_points(p1, vertex)
    v2_x, v2_y = vector_from_points(p2, vertex)
    len1 = vec_length(v1_x, v1_y)
    len2 = vec_length(v2_x, v2_y)
    return dot / (len1 * len2 + 1e-8)


def segment_ratio(p1: TorchPoint, p2: TorchPoint,
                  p3: TorchPoint, p4: TorchPoint) -> torch.Tensor:
    """|p1−p2| / |p3−p4|."""
    len1 = segment_length(p1, p2)
    len2 = segment_length(p3, p4)
    return len1 / (len2 + 1e-8)


def dist_to_line(point: TorchPoint, p1: TorchPoint,
                 p2: TorchPoint) -> torch.Tensor:
    """Absolute distance from *point* to line through *p1*, *p2*."""
    line = pp2lnf(p1, p2)
    return torch.abs(on_line(point, line))


def centroid_loss(centroid: TorchPoint, p1: TorchPoint,
                  p2: TorchPoint, p3: TorchPoint) -> torch.Tensor:
    """Squared error of *centroid* vs. true centroid of triangle."""
    expected_x = (p1.x + p2.x + p3.x) / 3
    expected_y = (p1.y + p2.y + p3.y) / 3
    return (centroid.x - expected_x) ** 2 + (centroid.y - expected_y) ** 2


def point_on_segment_loss(point: TorchPoint, p1: TorchPoint,
                          p2: TorchPoint) -> torch.Tensor:
    """Loss for *point* lying on segment p1–p2 (between p1 and p2)."""
    line = pp2lnf(p1, p2)
    on_line_loss = on_line(point, line) ** 2

    vec_x = p2.x - p1.x
    vec_y = p2.y - p1.y
    point_x = point.x - p1.x
    point_y = point.y - p1.y

    vec_len_sq = vec_x ** 2 + vec_y ** 2 + 1e-8
    t = (point_x * vec_x + point_y * vec_y) / vec_len_sq
    between_penalty = torch.relu(-t) + torch.relu(t - 1)
    return on_line_loss + 10.0 * between_penalty


def angle_bisector_equal_loss(p_vertex: TorchPoint, p1: TorchPoint,
                              p2: TorchPoint,
                              p_bisector: TorchPoint) -> torch.Tensor:
    """Loss: ∠(p1, vertex, bisector) == ∠(p2, vertex, bisector)."""
    cos1 = angle_cosine(p1, p_vertex, p_bisector)
    cos2 = angle_cosine(p2, p_vertex, p_bisector)
    return (cos1 - cos2) ** 2


def angle_diff_loss(p1: TorchPoint, p2: TorchPoint, p3: TorchPoint,
                    p4: TorchPoint, p5: TorchPoint,
                    p6: TorchPoint) -> torch.Tensor:
    """(cos ∠p1-p2-p3 − cos ∠p4-p5-p6)²."""
    cos1 = angle_cosine(p1, p2, p3)
    cos2 = angle_cosine(p4, p5, p6)
    return (cos1 - cos2) ** 2


def triangle_angle_difference_loss(pts: list, idx1: int,
                                   idx2: int) -> torch.Tensor:
    """Difference between two interior angles of a triangle."""
    v1_prev = pts[(idx1 - 1) % 3]
    v1_curr = pts[idx1]
    v1_next = pts[(idx1 + 1) % 3]

    v2_prev = pts[(idx2 - 1) % 3]
    v2_curr = pts[idx2]
    v2_next = pts[(idx2 + 1) % 3]

    cos1 = angle_cosine(v1_prev, v1_curr, v1_next)
    cos2 = angle_cosine(v2_prev, v2_curr, v2_next)
    return cos1 - cos2
