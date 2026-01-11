"""
Utility classes and functions for geometry solver
"""

import collections


FuncInfo = collections.namedtuple("FuncInfo", ["head", "args"])


def is_number(s):
    """Check if string represents a number"""
    try:
        float(s)
        return True
    except ValueError:
        return False
