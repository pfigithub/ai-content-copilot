
from datetime import datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.dependencies import get_current_user
from app.memory.commands import MemoryType
from app.memory.container import (
    get_memory_proposal_service,
    get_user_memory_store,
)
from app.memory.long_term_memory import UserMemoryStore
from app.memory.proposal_service import (
    MemoryProposalExpiredError,
    MemoryProposalNotFoundError,
    MemoryProposalService,
)


router = APIRouter(
    prefix="/api/v1/memory",
    tags=["memory"],
)


class MemoryProposalRequest(BaseModel):
    """Validate a proposed memory without accepting a client-supplied owner."""

    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1, max_length=128)
    content: str = Field(min_length=1, max_length=500)
    memory_type: MemoryType = "preference"

    @field_validator("memory_id", "content")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        """Normalize required text and reject whitespace-only values."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("This field cannot contain only whitespace.")
        return normalized


class MemoryProposalResponse(BaseModel):
    """Describe a pending proposal and its server-assigned owner."""

    proposal_id: str
    user_id: str
    memory_id: str
    content: str
    memory_type: MemoryType
    confirmation_required: bool
    expires_at: datetime


class MemoryActionResponse(BaseModel):
    """Describe the result of a memory confirmation, rejection, or deletion."""

    status: Literal["stored", "rejected", "deleted"]
    memory_id: str
    memory_type: MemoryType | None = None


class UserPreferencesResponse(BaseModel):
    """Return the authenticated user's saved preferences."""

    user_id: str
    preferences: list[dict]


@router.post(
    "/proposals",
    response_model=MemoryProposalResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_memory_proposal(
    request: MemoryProposalRequest,
    service: MemoryProposalService = Depends(get_memory_proposal_service),
    current_user: str = Depends(get_current_user),
) -> MemoryProposalResponse:
    """Create a pending memory proposal for the authenticated user."""
    try:
        pending = await service.propose(
            user_id=current_user,
            memory_id=request.memory_id,
            content=request.content,
            memory_type=request.memory_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    proposal = pending.proposal

    return MemoryProposalResponse(
        proposal_id=pending.proposal_id,
        user_id=current_user,
        memory_id=proposal.memory_id,
        content=proposal.content,
        memory_type=proposal.memory_type,
        confirmation_required=proposal.confirmation_required,
        expires_at=pending.expires_at,
    )


@router.post(
    "/proposals/{proposal_id}/confirm",
    response_model=MemoryActionResponse,
)
async def confirm_memory_proposal(
    proposal_id: Annotated[str, Path(min_length=1, max_length=64)],
    service: MemoryProposalService = Depends(get_memory_proposal_service),
    current_user: str = Depends(get_current_user),
) -> MemoryActionResponse:
    """Confirm a proposal only when it belongs to the authenticated user."""
    try:
        proposal = await service.confirm(
            proposal_id=proposal_id,
            user_id=current_user,
        )
    except MemoryProposalNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory proposal not found.",
        ) from exc
    except MemoryProposalExpiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Memory proposal has expired.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return MemoryActionResponse(
        status="stored",
        memory_id=proposal.memory_id,
        memory_type=proposal.memory_type,
    )


@router.post(
    "/proposals/{proposal_id}/reject",
    response_model=MemoryActionResponse,
)
async def reject_memory_proposal(
    proposal_id: Annotated[str, Path(min_length=1, max_length=64)],
    service: MemoryProposalService = Depends(get_memory_proposal_service),
    current_user: str = Depends(get_current_user),
) -> MemoryActionResponse:
    """Reject a proposal only when it belongs to the authenticated user."""
    try:
        proposal = await service.reject(
            proposal_id=proposal_id,
            user_id=current_user,
        )
    except MemoryProposalNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Memory proposal not found.",
        ) from exc
    except MemoryProposalExpiredError as exc:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Memory proposal has expired.",
        ) from exc

    return MemoryActionResponse(
        status="rejected",
        memory_id=proposal.memory_id,
        memory_type=proposal.memory_type,
    )


@router.get(
    "/preferences",
    response_model=UserPreferencesResponse,
)
async def list_user_preferences(
    store: UserMemoryStore = Depends(get_user_memory_store),
    current_user: str = Depends(get_current_user),
) -> UserPreferencesResponse:
    """List preferences in the authenticated user's memory namespace."""
    try:
        preferences = await store.get_preferences(user_id=current_user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return UserPreferencesResponse(
        user_id=current_user,
        preferences=preferences,
    )


@router.delete(
    "/memories/{memory_id}",
    response_model=MemoryActionResponse,
)
async def delete_user_memory(
    memory_id: Annotated[str, Path(min_length=1, max_length=128)],
    store: UserMemoryStore = Depends(get_user_memory_store),
    current_user: str = Depends(get_current_user),
) -> MemoryActionResponse:
    """Delete a memory only from the authenticated user's namespace."""
    try:
        await store.delete(
            user_id=current_user,
            memory_id=memory_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return MemoryActionResponse(
        status="deleted",
        memory_id=memory_id,
    )
