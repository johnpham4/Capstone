from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from uuid import uuid4
from pydantic import BaseModel, Field


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    request_id: str
    occurred_at: datetime = Field(default_factory=datetime.utcnow)


class UserInputReceived(Event):
    event_type: Literal["UserInputReceived"] = "UserInputReceived"
    user_input: str


class ModelProcessingCompleted(Event):
    event_type: Literal["ModelProcessingCompleted"] = "ModelProcessingCompleted"
    model_output: str
    dsl_commands: List[str]


class DiagramGenerationCompleted(Event):
    event_type: Literal["DiagramGenerationCompleted"] = "DiagramGenerationCompleted"
    diagram_path: str
    points: Dict[str, Any]


class ProcessingFailed(Event):
    event_type: Literal["ProcessingFailed"] = "ProcessingFailed"
    stage: Literal["input", "model", "diagram"]
    error_message: str
