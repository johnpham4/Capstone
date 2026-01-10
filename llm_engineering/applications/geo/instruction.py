"""
Instruction classes for geometry solver
"""


class Parameterize:
    """Instruction to parameterize geometric objects"""

    def __init__(self, objects, param_type, args=()):
        self.objects = objects
        self.param_type = param_type
        self.args = args

    def __str__(self):
        obj_str = ' '.join([str(o) for o in self.objects])
        if self.args:
            args_str = ' '.join([str(a) for a in self.args])
            return f"param ({obj_str}) ({self.param_type} {args_str})"
        else:
            return f"param ({obj_str}) {self.param_type}"


class Assert:
    """Instruction to assert a constraint"""

    def __init__(self, constraint):
        self.constraint = constraint

    def __str__(self):
        return f"assert ({self.constraint})"
