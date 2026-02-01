"""API schemas for orchestration endpoints."""

from pydantic import BaseModel, Field
from typing import Optional


class OrchestrateRequest(BaseModel):
    """Request schema for orchestrated geometry problem solving."""
    user_input: str = Field(..., description="User's question or problem in natural language")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    intent: Optional[str] = Field(None, description="Pre-classified intent: draw_only, solve_only, draw_and_solve, clarify")
    include_diagram: bool = Field(True, description="Whether to generate and include diagram")
    include_solution: bool = Field(True, description="Whether to generate and include solution")


class OrchestrateResponse(BaseModel):
    """Response schema for orchestration."""
    request_id: str
    session_id: str
    intent: str
    confidence: float

    # DSL and Diagram
    dsl: Optional[str] = None
    dsl_error: Optional[str] = None
    diagram_url: Optional[str] = None
    diagram_error: Optional[str] = None

    # Solution
    solution: Optional[str] = None
    solution_error: Optional[str] = None

    # Metadata
    processing_time_ms: Optional[float] = None
    steps_executed: Optional[list[str]] = None
