"""
DSL Instruction Models
Represents instructions parsed from DSL commands
"""
from dataclasses import dataclass
from typing import Any, Tuple
from llm_engineering.domains.geometry.types import DiagramType


@dataclass
class Parameter:
    """Represents a parameter declaration in DSL"""
    diagram_type: DiagramType
    objects: Tuple[Any, ...]
    param_type: str
    args: Tuple[Any, ...] = ()

    def __str__(self) -> str:
        obj_str = ' '.join([str(o) for o in self.objects])
        if self.args:
            args_str = ' '.join([str(a) for a in self.args])
            return f"param ({obj_str}) ({self.param_type} {args_str})"
        else:
            return f"param ({obj_str}) {self.param_type}"


@dataclass
class Assertion:
    """Represents an assertion/constraint in DSL"""
    constraint_type: str = None
    objects: Tuple[Any, ...] = ()
    constraint: Any = None

    def __str__(self) -> str:
        if self.constraint_type:
            obj_str = ' '.join([str(o) for o in self.objects])
            return f"assert ({self.constraint_type} {obj_str})"
        return f"assert ({self.constraint})"


@dataclass
class Definition:
    """Represents a definition of derived object in DSL"""
    obj_name: str
    obj_type: str
    computation: Any

    def __str__(self) -> str:
        return f"define {self.obj_name} {self.obj_type} ({self.computation})"
