"""API schemas for orchestration endpoints."""

from pydantic import BaseModel, Field
from typing import Optional


class OrchestrateRequest(BaseModel):
    user_input: str = Field(..., description="User's question or problem in natural language")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    intent: Optional[str] = Field(None, description="Pre-classified intent: draw_only, solve_only, draw_and_solve, clarify")
    include_diagram: bool = Field(True, description="Whether to generate and include diagram")
    include_solution: bool = Field(True, description="Whether to generate and include solution")


class OrchestrateResponse(BaseModel):
    request_id: str
    session_id: str
    status: str = "queued"
    celery_task_id: Optional[str] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None

    dsl: Optional[str] = None
    dsl_error: Optional[str] = None
    diagram_url: Optional[str] = None
    diagram_error: Optional[str] = None

    solution: Optional[str] = None
    solution_error: Optional[str] = None

    processing_time_ms: Optional[float] = None
    steps_executed: Optional[list[str]] = None

