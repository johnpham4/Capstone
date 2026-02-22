"""Orchestration DTOs."""

from pydantic import BaseModel, Field
from typing import Literal


class OrchestrationRequest(BaseModel):
    user_input: str = Field(..., description="User's natural language query")
    mode: Literal["auto", "diagram", "solve", "both"] = Field(
        default="auto",
        description="Execution mode: auto (detect), diagram (draw only), solve (math only), both (parallel)",
    )
    llm_mock: bool = Field(
        default=False,
        description="Skip SageMaker LLM call and return canned DSL for testing",
    )
