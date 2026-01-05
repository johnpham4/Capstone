"""
Simple Geometry Builder - Triangles and Lines Only

A simplified geometry diagram builder supporting only triangles and lines,
rewritten from the original geo-model-builder.

Example usage:
    from geo_building import build

    gmbl_code = '''
    (param (A B C) triangle)
    (define M point (midp B C))
    '''

    diagram = build(gmbl_code.split('\\n'), show_plot=True)
"""

from .builder import build, build_from_file, SimpleBuilder
from .diagram import Diagram
from .primitives import Point, Line, Num
from .instruction_reader import InstructionReader

__version__ = "1.0.0"
__all__ = [
    'build',
    'build_from_file',
    'SimpleBuilder',
    'Diagram',
    'Point',
    'Line',
    'Num',
    'InstructionReader'
]
