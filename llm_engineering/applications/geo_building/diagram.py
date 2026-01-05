"""Simple diagram plotter for triangles and lines"""
import collections
import matplotlib.pyplot as plt
import os
import numpy as np
import math


UNNAMED_ALPHA = 0.1
MIN_AXIS_VAL = -10
MAX_AXIS_VAL = 10


class Diagram(collections.namedtuple("Diagram", [
    "named_points", "named_lines", "segments", "seg_colors",
    "unnamed_points", "unnamed_lines", "ndgs", "goals"
])):
    def plot(self, show=True, save=False, fname=None, return_fig=False, show_unnamed=True):
        """Plot the geometric diagram"""
        unnamed_points = self.unnamed_points if show_unnamed else list()
        unnamed_lines = self.unnamed_lines if show_unnamed else list()

        # Plot named points
        xs = [p.x for p in self.named_points.values()]
        ys = [p.y for p in self.named_points.values()]
        names = [n for n in self.named_points.keys()]

        fig, ax = plt.subplots()

        if xs and ys:
            ax.scatter(xs, ys)
            for i, n in enumerate(names):
                ax.annotate(str(n), (xs[i], ys[i]))

        # Plot unnamed points
        if unnamed_points:
            u_xs = [p.x for p in unnamed_points]
            u_ys = [p.y for p in unnamed_points]
            ax.scatter(u_xs, u_ys, c="black", alpha=UNNAMED_ALPHA)

        # Plot segments (triangle edges)
        for (p1, p2), c in zip(self.segments, self.seg_colors):
            plt.plot([p1.x, p2.x], [p1.y, p2.y], c=c)

        plt.axis('scaled')
        plt.axis('square')

        # Set axis limits
        have_points = self.named_points or unnamed_points
        if not have_points:
            lo_x_lim, lo_y_lim = -2, -2
            hi_x_lim, hi_y_lim = 2, 2
        else:
            (lo_x_lim, hi_x_lim) = ax.get_xlim()
            (lo_y_lim, hi_y_lim) = ax.get_ylim()
            if self.named_lines:
                lo_x_lim -= 1
                hi_x_lim += 1
                lo_y_lim -= 1
                hi_y_lim += 1
            lo_x_lim = max(MIN_AXIS_VAL, lo_x_lim)
            hi_x_lim = min(MAX_AXIS_VAL, hi_x_lim)
            lo_y_lim = max(MIN_AXIS_VAL, lo_y_lim)
            hi_y_lim = min(MAX_AXIS_VAL, hi_y_lim)

        ax.set_xlim([lo_x_lim, hi_x_lim])
        ax.set_ylim([lo_y_lim, hi_y_lim])

        def plot_line(L, name=None):
            """Plot a line given in normal form"""
            (nx, ny), r = L
            if nx == 0:
                l_angle = math.pi / 2
            else:
                l_angle = math.atan(ny / nx) % math.pi

            if l_angle == 0:
                if name is not None:
                    plt.axvline(x=r, label=name)
                else:
                    plt.axvline(x=r, c="black", alpha=UNNAMED_ALPHA)
            else:
                slope = -1 / math.tan(l_angle)
                intercept = r / math.sin(l_angle)
                (lo_x_lim, hi_x_lim) = ax.get_xlim()
                x_vals = np.array((lo_x_lim - 0.2, hi_x_lim + 0.2))
                y_vals = intercept + slope * x_vals
                if name is not None:
                    plt.plot(x_vals, y_vals, label=name)
                else:
                    plt.plot(x_vals, y_vals, c="black", alpha=UNNAMED_ALPHA)

        # Plot unnamed lines
        for L in unnamed_lines:
            plot_line(L)

        # Plot named lines
        for l, L in self.named_lines.items():
            l_name = l.val
            plot_line(L, l_name)

        if self.named_lines:
            plt.legend()

        if return_fig:
            return plt

        if show:
            plt.show()
        if save:
            if fname is None:
                raise RuntimeError("Must supply filename if saving plot")
            if os.path.isfile(fname):
                raise RuntimeError(f"File {fname} already exists")
            plt.savefig(fname)
            plt.close()
