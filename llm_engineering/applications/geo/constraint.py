"""
Constraint representation for geometry problems
"""


class Constraint:
    """Represents a geometric constraint"""

    def __init__(self, pred, args, negate=False):
        self.pred = pred
        self.args = args
        self.negate = negate

    def __str__(self):
        c_str = ' '.join([self.pred] + [str(a) for a in self.args])
        if self.negate:
            return f"not ({c_str})"
        else:
            return c_str
