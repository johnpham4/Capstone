"""Domain models package - Core business entities and value objects.

Organization:
├── geometry/        - Geometric shapes, points, diagrams
├── orchestration/   - Agent workflow and state management
├── training/        - Dataset, documents, prompts for model training
├── shared/          - Common types, exceptions
├── events.py        - Domain events
├── inference.py     - Inference abstractions
└── processing_request.py - Request processing entities
"""

# Geometry domain
from .geometry import (
    GeometricPoint,
    Diagram,
    Point,
    Line,
    Triangle,
    Circle,
    DiagramType,
    TriangleType,
    QuadrilateralType,
    CircleType,
)

# Training domain
from .training import (
    InstructDataset,
    InstructDatasetSample,
    TrainTestSplit,
    Document,
    Prompt,
)

# Shared utilities
from .shared import (
    ImproperlyConfigured,
    DomainError,
    DataCategory,
)

# Standalone models
from .queue.events import Event

__all__ = [
    # Geometry
    "GeometricPoint",
    "Diagram",
    "Point",
    "Line",
    "Triangle",
    "Circle",
    "DiagramType",
    "TriangleType",
    "QuadrilateralType",
    "CircleType",
    # Orchestration
    "AgentState",
    "Intent",
    "Message",
    # Training
    "InstructDataset",
    "InstructDatasetSample",
    "TrainTestSplit",
    "Document",
    "Prompt",
    # Shared
    "ImproperlyConfigured",
    "DomainError",
    "DataCategory",
    # Standalone
    "Event",
    "ProcessingRequest",
    "RequestStatus",
]


