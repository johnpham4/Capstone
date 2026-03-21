from pydantic import BaseModel, Field
from typing import Literal


class RewriteResponse(BaseModel):
    problem_statement: str = Field(description="The clean geometry problem extracted from user input")
    mode: Literal["diagram", "both"] = Field(default="diagram", description="diagram = draw only, both = draw + solve")


class OrchestrationRequest(BaseModel):
    user_input: str
    mode: Literal["diagram", "both"] = Field(default="diagram")
    llm_mock: bool = Field(default=False)


class StreamOrchestrationRequest(BaseModel):
    user_input: str
    mode: Literal["diagram", "both"] = Field(default="diagram")
    llm_mock: bool = Field(default=False)
