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
========
"""
DSL Instruction Models - Domain entities using dataclass.
Represents instructions parsed from DSL commands.
"""
from dataclasses import dataclass
from typing import Any, Tuple
from src.services.diagram.models.types import DiagramType


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
>>>>>>>> 6cf03dda8dad8bb8fa1226b8b4e9166c3f287527:src/services/diagram/models/instructions.py
