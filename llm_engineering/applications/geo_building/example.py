"""Example usage of the simple geometry builder"""
from builder import build


# Example 1: Simple equilateral triangle
example1 = """
(param (A B C) triangle)
"""

# Example 2: Right triangle with midpoint
example2 = """
(param (A B C) (right-tri A))
(define M point (midp B C))
"""

# Example 3: Triangle with incenter
example3 = """
(param (A B C) triangle)
(define I point (incenter A B C))
"""

# Example 4: Triangle with perpendicular bisector
example4 = """
(param (A B C) triangle)
(define M point (midp B C))
(define l line (mediator B C))
"""

# Example 5: Triangle with angle bisector
example5 = """
(param (A B C) triangle)
(define l line (i-bisector A B C))
"""

# Example 6: Two triangles with connecting line
example6 = """
(param (A B C) triangle)
(define D point coords)
(define l line (connecting A D))
"""

# Example 7: Triangle with centroid
example7 = """
(param (A B C) triangle)
(define G point (centroid A B C))
"""

# Example 8: Isosceles triangle
example8 = """
(param (A B C) (iso-tri A))
(define M point (midp B C))
"""

if __name__ == "__main__":
    print("Example 1: Simple equilateral triangle")
    build(example1.strip().split('\n'), show_plot=True)

    print("\nExample 2: Right triangle with midpoint")
    build(example2.strip().split('\n'), show_plot=True)

    print("\nExample 3: Triangle with incenter")
    build(example3.strip().split('\n'), show_plot=True)

    print("\nExample 4: Triangle with perpendicular bisector")
    build(example4.strip().split('\n'), show_plot=True)

    print("\nExample 7: Triangle with centroid")
    build(example7.strip().split('\n'), show_plot=True)
