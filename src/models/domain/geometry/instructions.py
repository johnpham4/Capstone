<<<<<<<< HEAD:src/models/domain/geometry/instructions.py
"""
DSL Instruction Models - Domain entities using dataclass.
Represents instructions parsed from DSL commands.
"""
from dataclasses import dataclass
from typing import Any, Tuple
from src.models.domain.geometry.types import DiagramType


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
    constraint_type: str = None
    objects: Tuple[Any, ...] = ()
    value: float = None
    constraint: Any = None
    distance: Any = None  

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


@dataclass
class DistanceValue:
    """Wrapper for distance value in constraints"""
    val: float
    
    def __str__(self) -> str:
        return str(self.val)
========
"""
DSL Instruction Models - Domain entities using dataclass.
Represents instructions parsed from DSL commands.
"""
from dataclasses import dataclass
from typing import Any, Tuple
from .types import DiagramType


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
    constraint_type: str = None
    objects: Tuple[Any, ...] = ()
    value: float = None
    constraint: Any = None
    distance: Any = None  

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


@dataclass
class DistanceValue:
    """Wrapper for distance value in constraints"""
    val: float
    
    def __str__(self) -> str:
        return str(self.val)
>>>>>>>> minh-re:src/services/diagram/model/instructions.py
