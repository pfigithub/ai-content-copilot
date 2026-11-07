
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.memory.commands import (
    MemoryCommandService,
    MemoryProposal,
    MemoryType,
)
from app.memory.proposal_repository import (
    MemoryProposalExpiredError,
    MemoryProposalNotFoundError,
    MemoryProposalRepository,
    PendingMemoryProposal,
)


class MemoryProposalService:
    """Coordinate proposal creation and explicit memory decisions."""

    def __init__(
        self,
        command_service: MemoryCommandService,
        repository: MemoryProposalRepository,
        ttl_seconds: int = 600,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero.")

        self.command_service = command_service
        self.repository = repository
        self.ttl_seconds = ttl_seconds

    async def propose(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        memory_type: MemoryType = "preference",
    ) -> PendingMemoryProposal:
        """Create and persist a proposal without writing long-term memory."""
        proposal = self.command_service.propose(
            user_id=user_id,
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
        )

        now = datetime.now(timezone.utc)

        pending = PendingMemoryProposal(
            proposal_id=str(uuid4()),
            proposal=proposal,
            expires_at=now + timedelta(seconds=self.ttl_seconds),
        )

        await self.repository.create(pending)
        return pending

    async def confirm(
        self,
        proposal_id: str,
        user_id: str,
    ) -> MemoryProposal:
        """Persist a confirmed proposal and consume it exactly once."""
        return await self.repository.confirm(
            proposal_id=proposal_id,
            user_id=user_id,
            on_confirm=self.command_service.confirm,
        )

    async def reject(
        self,
        proposal_id: str,
        user_id: str,
    ) -> MemoryProposal:
        """Consume a proposal after the authenticated owner rejects it."""
        return await self.repository.reject(
            proposal_id=proposal_id,
            user_id=user_id,
        )
