"""Mock DSL samples for profiling the geometry diagram pipeline.

These samples exercise increasing complexity: simple triangles, quadrilaterals,
circles, midpoints, projections, and combined constraints. They are designed to
mimic real model output without requiring any LLM inference.
"""

MOCK_DSLS: list[str] = [
    # 1. Simple right triangle
    """(triangle (A B C) (right B))""",

    # 2. Right triangle with midpoint segment
    """(triangle (A B C) (right B))
(define D point (midpoint A B))
(segment D E)""",

    # 3. Isosceles triangle
    """(triangle (A B C) (isosceles A))""",

    # 4. Equilateral triangle
    """(triangle (A B C) (equilateral))""",

    # 5. Square
    """(square (A B C D))""",

    # 6. Rectangle
    """(rectangle (A B C D))""",

    # 7. Parallelogram
    """(parallelogram (A B C D))""",

    # 8. Rhombus
    """(rhombus (A B C D))""",

    # 9. Right triangle with altitude projection
    """(triangle (A B C) (right A))
(define H point (projection A (segment B C)))
(segment A H)""",

    # 10. Triangle with midpoint theorem segment
    """(triangle (A B C))
(define D point (midpoint A B))
(define E point (midpoint A C))
(segment D E)""",

    # 11. Circle with inscribed triangle
    """(circle O (circumcircle A B C))
(triangle (A B C))""",

    # 12. Circle with tangent line
    """(circle O (radius 1.0))
(define M point (on-circle M O))
(tangent M (circle O) AB)""",

    # 13. Triangle with centroid
    """(triangle (A B C))
(define G point (centroid A B C))""",

    # 14. Triangle with orthocenter
    """(triangle (A B C))
(define H point (orthocenter A B C))""",

    # 15. Right triangle with circumcenter (Thales)
    """(triangle (A B C) (right C))
(define O point (circumcenter A B C))""",

    # 16. Quadrilateral with diagonal intersection
    """(quadrilateral (A B C D))
(define I point (intersection (segment A C) (segment B D)))""",

    # 17. Isosceles triangle with angle bisector
    """(triangle (A B C) (isosceles A))
(define D point (angle-bisector A B C))
(segment A D)""",

    # 18. Triangle with incircle
    """(triangle (A B C))
(circle I (incircle A B C))""",

    # 19. Parallel/perpendicular constraints
    """(triangle (A B C))
(define D point (midpoint A B))
(define E point (midpoint A C))
(segment D E)
(perpendicular (segment D E) (segment B C))""",

    # 20. Complex combined diagram
    """(triangle (A B C) (right B))
(define D point (midpoint A B))
(define E point (midpoint A C))
(segment D E)
(circle O (circumcircle A B C))
(define H point (orthocenter A B C))""",
]


def get_mock_dsls(count: int | None = None) -> list[str]:
    """Return a subset or all mock DSLs."""
    if count is None:
        return MOCK_DSLS[:]
    return MOCK_DSLS[:count]
