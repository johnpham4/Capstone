"""
Diagram plotting utilities for geometry solver
Simplified version for triangles, lines, and points
"""

import matplotlib.pyplot as plt
import numpy as np
import math


class Diagram:
    """Simple diagram class for visualizing geometry problems"""

    def __init__(self):
        self.points = {}  # name -> Point
        self.triangles = []  # list of (p1, p2, p3)
        self.segments = []  # list of (p1, p2, color)
        self.lines = {}  # name -> Line

    def add_point(self, name, point):
        """Add a named point to the diagram"""
        self.points[name] = point

    def add_triangle(self, p1, p2, p3):
        """Add a triangle to the diagram"""
        self.triangles.append((p1, p2, p3))

    def add_segment(self, p1, p2, color='blue'):
        """Add a line segment to the diagram"""
        self.segments.append((p1, p2, color))

    def add_line(self, name, line):
        """Add a named line to the diagram"""
        self.lines[name] = line

    def plot(self, show=True, save=False, filename=None, title="Geometry Solution"):
        """Plot the diagram using matplotlib"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot points
        if self.points:
            xs = [p.x for p in self.points.values()]
            ys = [p.y for p in self.points.values()]
            names = list(self.points.keys())

            ax.scatter(xs, ys, s=100, c='red', zorder=5)
            for name, x, y in zip(names, xs, ys):
                ax.annotate(name, (x, y), xytext=(5, 5),
                           textcoords='offset points', fontsize=12, fontweight='bold')

        # Plot triangles (as filled polygons)
        for p1, p2, p3 in self.triangles:
            xs = [p1.x, p2.x, p3.x, p1.x]
            ys = [p1.y, p2.y, p3.y, p1.y]
            ax.fill(xs, ys, alpha=0.2, color='lightblue', edgecolor='blue', linewidth=2)

        # Plot segments
        for p1, p2, color in self.segments:
            ax.plot([p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=2, marker='o')

        # Set axis properties
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Auto-scale with some padding
        if self.points:
            all_xs = [p.x for p in self.points.values()]
            all_ys = [p.y for p in self.points.values()]

            x_min, x_max = min(all_xs), max(all_xs)
            y_min, y_max = min(all_ys), max(all_ys)

            # Add 20% padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            padding = max(x_range, y_range) * 0.2 + 0.5

            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)

        plt.tight_layout()

        if save and filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n✓ Diagram saved to {filename}")

        if show:
            plt.show()

        return fig, ax
