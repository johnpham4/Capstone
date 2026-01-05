"""Instructions for geometry construction"""
import collections


class Assert:
    def __init__(self, constraint):
        self.constraint = constraint

    def __str__(self):
        return f"assert ({self.constraint})"


class AssertNDG:
    def __init__(self, constraint):
        self.constraint = constraint

    def __str__(self):
        return f"assertNDG ({self.constraint})"


class Compute:
    def __init__(self, obj_name, computation):
        self.obj_name = obj_name
        self.computation = computation

    def __str__(self):
        if isinstance(self.computation, collections.abc.Iterable) and type(self.computation) != str:
            comp_str = f"{self.computation.val[0]} {' '.join(str(x) for x in self.computation.val[1])}"
        else:
            comp_str = str(self.computation)
        return f"define {self.obj_name} ({comp_str})"


class Eval:
    def __init__(self, constraint):
        self.constraint = constraint

    def __str__(self):
        return f"eval ({self.constraint})"


class Parameterize:
    def __init__(self, obj_name, parameterization):
        self.obj_name = obj_name
        self.parameterization = parameterization

    def __str__(self):
        if self.parameterization[0] in ["coords", "line"]:
            param_str = self.parameterization[0]
        elif isinstance(self.parameterization, collections.abc.Iterable) and type(self.parameterization) != str:
            param_str = f"({self.parameterization[0]} {' '.join(str(x) for x in self.parameterization[1])})"
        else:
            param_str = str(self.parameterization)
        return f"parameterize {self.obj_name} {param_str}"


class Sample:
    def __init__(self, points, sampler, args=()):
        self.points = points
        self.sampler = sampler
        self.args = args

    def __str__(self):
        return f"sample ({' '.join([str(p) for p in self.points])}) {self.sampler} ({' '.join([str(a) for a in self.args])})"
