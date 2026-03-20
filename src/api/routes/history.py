from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user
from src.infrastructures.database.session import get_db
from src.models.dto.user import User
from src.models.dto.history import HistoryDetail, PaginatedHistory
from src.services.history import HistoryService

router = APIRouter(prefix="/api/v1/history")


@router.get("", response_model=PaginatedHistory)
async def list_history(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
):
    """Return paginated history of the authenticated user's requests."""
    service = HistoryService(db)
    return await service.list_history(
        user_id=current_user.id,
        page=page,
        page_size=page_size,
    )


@router.get("/{request_id}", response_model=HistoryDetail)
async def get_history_detail(
    request_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Return full detail for a single request including diagram & solution."""
    service = HistoryService(db)
    detail = await service.get_detail(request_id)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found",
        )
    return detail


@router.delete("/{request_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_history_item(
    request_id: str,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
):
    """Delete a request and cascaded diagram/solution if owned by user."""
    service = HistoryService(db)
    deleted = await service.delete_request(request_id, user_id=current_user.id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Request not found or not owned by you",
        )
