from typing import Any, Literal

from pydantic import BaseModel, Field


Mode = Literal["diagram", "solve", "both"]


class RewriteResponse(BaseModel):
    problem_statement: str 
    mode: Mode = Field(default="diagram")


class OrchestrationRequest(BaseModel):
    user_input: str
    mode: Mode = Field(default="diagram")
    llm_mock: bool = Field(default=False)


class StreamOrchestrationRequest(BaseModel):
    user_input: str
    mode: Mode = Field(default="diagram")
    llm_mock: bool = Field(default=False)


class OrchestrationResponse(BaseModel):
    status: Literal["success"]
    request_id: str
    mode: Mode
    diagram: dict[str, Any] | None = None
    solution: dict[str, Any] | None = None
