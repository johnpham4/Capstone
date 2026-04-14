from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.rate_limiter import rate_limit_orchestration
from src.infrastructures.database.session import get_db
from src.models.dto.user import User
from src.models.dto.orchestration import OrchestrationRequest, OrchestrationResponse
from src.services.orchestration import OrchestrationService, OrchestrationError
from src.services.history import HistoryService
from src.prompts import DSL_INFERENCE_INSTRUCTION


router = APIRouter()

_orchestration_service = OrchestrationService(diagram_prompt=DSL_INFERENCE_INSTRUCTION)


@router.post("/api/v1/orchestration", response_model=OrchestrationResponse)
async def execute_orchestration(
    request: OrchestrationRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    _rate: None = Depends(rate_limit_orchestration),
):
    try:
        history = HistoryService(db)
        return await _orchestration_service.execute(
            user_id=current_user.id,
            user_input=request.user_input,
            mode=request.mode,
            history=history,
            llm_mock=request.llm_mock,
            image_base64=request.image_base64,
        )

    except OrchestrationError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": exc.code,
                "message": exc.message,
                "request_id": exc.request_id,
            },
        )

