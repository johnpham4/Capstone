"""Simple test runner for Asymptote DSL examples"""
from asymptote_builder import AsymptoteBuilder


# Example 1: Tam giác cân với trực tâm
print("=" * 60)
print("Example 1: Isosceles Triangle with Orthocenter")
print("=" * 60)

example1 = """
(triangle A B C (isosceles A))
(point P (orthocenter A B C))
"""

try:
    builder1 = AsymptoteBuilder(example1.strip().split('\n'), optimize=True, n_iterations=500)
    diagram1 = builder1.build()
    print("✓ Example 1: Build successful!")
    print(f"  Points: {list(diagram1.named_points.keys())}")
    print(f"  Coordinates:")
    for p, sp in diagram1.named_points.items():
        print(f"    {p}: ({sp.x:.3f}, {sp.y:.3f})")

    # Vẽ và lưu hình
    diagram1.plot(show=False, save=True, fname='example_1.png')
    print("  ✓ Saved to example_1.png")
except Exception as e:
    print(f"✗ Example 1 failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Example 2: Tam giác vuông với các ràng buộc
print("=" * 60)
print("Example 2: Right Triangle with Constraints")
print("=" * 60)

example2 = """
(triangle A B C (right-at B))
(free-point D (on-segment A B))
(free-point E (on-segment A C))

(line l_BC (connecting B C))
(line l_DE (connecting D E))

(assert (parallel l_BC l_DE))
(assert (right-angle A D E))
"""

try:
    builder2 = AsymptoteBuilder(example2.strip().split('\n'), optimize=True, n_iterations=1000)
    diagram2 = builder2.build()
    print("✓ Example 2: Build successful!")
    print(f"  Points: {list(diagram2.named_points.keys())}")

    # Vẽ và lưu hình
    diagram2.plot(show=False, save=True, fname='example_2.png')
    print("  ✓ Saved to example_2.png")
except Exception as e:
    print(f"✗ Example 2 failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Example 3: Đường trung bình tam giác
print("=" * 60)
print("Example 3: Triangle Midpoint Theorem")
print("=" * 60)

example3 = """
(triangle A B C)
(point D (midpoint A B))
(point E (midpoint A C))

(line l_BC (connecting B C))
(line l_DE (connecting D E))

(assert (parallel l_BC l_DE))
"""

try:
    builder3 = AsymptoteBuilder(example3.strip().split('\n'), optimize=True, n_iterations=500)
    diagram3 = builder3.build()
    print("✓ Example 3: Build successful!")

    # Vẽ và lưu hình
    diagram3.plot(show=False, save=True, fname='example_3.png')
    print("  ✓ Saved to example_3.png")
    print(f"  Points: {list(diagram3.named_points.keys())}")
    print(f"  Coordinates:")
    for p, sp in diagram3.named_points.items():
        print(f"    {p}: ({sp.x:.3f}, {sp.y:.3f})")
except Exception as e:
    print(f"✗ Example 3 failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Example 4: Tam giác với phân giác
print("=" * 60)
print("Example 4: Right Triangle with Angle Bisector")
print("=" * 60)

example4 = """
(triangle A B C (right-at B))
(point D (midpoint A B))
(point E (midpoint A C))
(free-point F (on-segment B C))

(line l_BC (connecting B C))
(line l_DE (connecting D E))

(assert (parallel l_BC l_DE))
(assert (equal (angle B A F) (angle C A F)))
(assert (right-angle A D E))
"""

try:
    builder4 = AsymptoteBuilder(example4.strip().split('\n'), optimize=True, n_iterations=1000)
    diagram4 = builder4.build()
    print("✓ Example 4: Build successful!")
    print(f"  Points: {list(diagram4.named_points.keys())}")

    # Vẽ và lưu hình
    diagram4.plot(show=False, save=True, fname='example_4.png')
    print("  ✓ Saved to example_4.png")
except Exception as e:
    print(f"✗ Example 4 failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("All examples completed!")
print("=" * 60)
