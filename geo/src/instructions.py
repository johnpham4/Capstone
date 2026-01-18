class parameter: # tạo instruction object
    def __init__(self, objects, param_type, args=()):
        self.objects = objects  # list of Primitive objects
        self.param_type = param_type  # str, example: "iso-tri", "circle", etc.
        self.args = args  # additional arguments if any
        
    def __str__(self):
        object_str = ' '.join([str(obj) for obj in self.objects])
        if self.args():
            args_str = ' '.join([str(a) for a in self.args])
            return f"(param ({object_str}) {self.param_type} {args_str})"
        else:
            return f"(param ({object_str}) {self.param_type})"