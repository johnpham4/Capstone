"""Utility functions and classes for geometry builder"""
import collections
import random
import string


class Root(collections.namedtuple("Root", ["pred", "vars"])):
    def __str__(self):
        if self.pred == "arbitrary":
            return "root-arbitrary"
        else:
            return f"(root-{self.pred} {' '.join([str(v) for v in self.vars])})"


FuncInfo = collections.namedtuple("FuncInfo", ["head", "args"])


DEFAULTS = {
    "decay_steps": 1e3,
    "decay_rate": 0.7,
    "eps": 1e-3,
    "learning_rate": 1e-1,
    "loss_freq": 100,
    "make_distinct": 1e-2,
    "min_dist": 0.1,
    "n_iterations": 5000,
    "plot_freq": 1000,
    "regularize_points": 1e-6,
    "n_models": 1,
    "n_tries": 3,
    "n_inits": 10,
    "verbosity": 0
}


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str
