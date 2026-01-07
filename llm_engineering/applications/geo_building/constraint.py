"""Geometric constraints for triangles and lines"""


class Constraint:
    def __init__(self, pred, args, negate):
        self.pred = pred
        self.args = args
        self.negate = negate

    def ndgs(self):
        """Non-degeneracy conditions"""
        if self.pred == "i-bisector" and not self.negate:
            return [Constraint("coll", self.args[1:], False)]
        else:
            return list()

    def orders(self):
        """Ordering constraints"""
        if self.pred == "i-bisector" and not self.negate:
            x, b, a, c = self.args[0], self.args[1], self.args[2], self.args[3]
            c1 = Constraint("same-side", [x, b, a, c], False)
            c2 = Constraint("same-side", [x, c, a, b], False)
            return [c1, c2]
        else:
            return list()

    def __str__(self):
        c_str = ' '.join([self.pred] + [str(a) for a in self.args])
        if self.negate:
            return f"not ({c_str})"
        else:
            return c_str
