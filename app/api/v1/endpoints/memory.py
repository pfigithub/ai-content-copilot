from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

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
    user_id: str = Field(
        min_length=1,
        max_length=128,
    )
    memory_id: str = Field(
        min_length=1,
        max_length=128,
    )
    content: str = Field(
        min_length=1,
        max_length=500,
    )
    memory_type: MemoryType = "preference"


class MemoryDecisionRequest(BaseModel):
    user_id: str = Field(
        min_length=1,
        max_length=128,
    )


class MemoryProposalResponse(BaseModel):
    proposal_id: str
    user_id: str
    memory_id: str
    content: str
    memory_type: MemoryType
    confirmation_required: bool
    expires_at: datetime


class MemoryActionResponse(BaseModel):
    status: Literal[
        "stored",
        "rejected",
        "deleted",
    ]
    memory_id: str
    memory_type: MemoryType | None = None


class UserPreferencesResponse(BaseModel):
    user_id: str
    preferences: list[dict]


@router.post(
    "/proposals",
    response_model=MemoryProposalResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_memory_proposal(
    request: MemoryProposalRequest,
    service: MemoryProposalService = Depends(
        get_memory_proposal_service
    ),
):
    try:
        pending = service.propose(
            user_id=request.user_id,
            memory_id=request.memory_id,
            content=request.content,
            memory_type=request.memory_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    proposal = pending.proposal

    return MemoryProposalResponse(
        proposal_id=pending.proposal_id,
        user_id=proposal.user_id,
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
    proposal_id: str,
    request: MemoryDecisionRequest,
    service: MemoryProposalService = Depends(
        get_memory_proposal_service
    ),
):
    try:
        proposal = await service.confirm(
            proposal_id=proposal_id,
            user_id=request.user_id,
        )
    except MemoryProposalNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Memory proposal not found.",
        ) from exc
    except MemoryProposalExpiredError as exc:
        raise HTTPException(
            status_code=410,
            detail="Memory proposal has expired.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
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
def reject_memory_proposal(
    proposal_id: str,
    request: MemoryDecisionRequest,
    service: MemoryProposalService = Depends(
        get_memory_proposal_service
    ),
):
    try:
        proposal = service.reject(
            proposal_id=proposal_id,
            user_id=request.user_id,
        )
    except MemoryProposalNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Memory proposal not found.",
        ) from exc
    except MemoryProposalExpiredError as exc:
        raise HTTPException(
            status_code=410,
            detail="Memory proposal has expired.",
        ) from exc

    return MemoryActionResponse(
        status="rejected",
        memory_id=proposal.memory_id,
        memory_type=proposal.memory_type,
    )


@router.get(
    "/users/{user_id}/preferences",
    response_model=UserPreferencesResponse,
)
async def list_user_preferences(
    user_id: str,
    store: UserMemoryStore = Depends(
        get_user_memory_store
    ),
):
    try:
        preferences = await store.get_preferences(
            user_id=user_id
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return UserPreferencesResponse(
        user_id=user_id,
        preferences=preferences,
    )


@router.delete(
    "/users/{user_id}/memories/{memory_id}",
    response_model=MemoryActionResponse,
)
async def delete_user_memory(
    user_id: str,
    memory_id: str,
    store: UserMemoryStore = Depends(
        get_user_memory_store
    ),
):
    try:
        await store.delete(
            user_id=user_id,
            memory_id=memory_id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return MemoryActionResponse(
        status="deleted",
        memory_id=memory_id,
    )