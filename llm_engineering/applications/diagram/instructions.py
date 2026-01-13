class Parameter:
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