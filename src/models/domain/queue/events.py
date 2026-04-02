from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional


class EventType(str, Enum):
    TASK_QUEUED = "task_queued"
    INTENT_CLASSIFIED = "intent_classified"
    DSL_GENERATED = "dsl_generated"
    DIAGRAM_READY = "diagram_ready"
    SOLUTION_COMPLETE = "solution_complete"
    TASK_FAILED = "task_failed"


class Event(BaseModel):
    event_type: EventType
    task_id: str
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
