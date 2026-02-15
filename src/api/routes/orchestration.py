from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

from src.services.orchestration import Orchestrator
from src.api.routes.prompt import INSTRUCTION_PROMPT


router = APIRouter()
orchestrator = Orchestrator(diagram_prompt=INSTRUCTION_PROMPT)

class OrchestrationRequest(BaseModel):
    user_input: str = Field(..., description="User's natural language query")
    mode: Literal["auto", "diagram", "solve", "both"] = Field(
        default="auto",
        description="Execution mode: auto (detect), diagram (draw only), solve (math only), both (parallel)"
    )


@router.post("/api/v1/orchestration")
async def execute_orchestration(request: OrchestrationRequest):
    try:
        result = await orchestrator.execute(request.user_input, mode=request.mode)

        return {
            "status": "success",
            "mode": result.get("mode", request.mode),
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
