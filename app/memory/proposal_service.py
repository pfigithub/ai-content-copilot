from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app.memory.commands import (
    MemoryCommandService,
    MemoryProposal,
    MemoryType,
)


class MemoryProposalNotFoundError(Exception):
    pass


class MemoryProposalExpiredError(Exception):
    pass


@dataclass(frozen=True)
class PendingMemoryProposal:
    proposal_id: str
    proposal: MemoryProposal
    expires_at: datetime


class MemoryProposalService:
    def __init__(
        self,
        command_service: MemoryCommandService,
        ttl_seconds: int = 600,
    ):
        if ttl_seconds <= 0:
            raise ValueError(
                "ttl_seconds must be greater than zero."
            )

        self.command_service = command_service
        self.ttl_seconds = ttl_seconds
        self._pending: dict[
            str, PendingMemoryProposal
        ] = {}

    def propose(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        memory_type: MemoryType = "preference",
    ) -> PendingMemoryProposal:
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
            expires_at=now + timedelta(
                seconds=self.ttl_seconds
            ),
        )

        self._pending[pending.proposal_id] = pending

        return pending

    def _get_owned_proposal(
        self,
        proposal_id: str,
        user_id: str,
    ) -> PendingMemoryProposal:
        pending = self._pending.get(proposal_id)

        if pending is None:
            raise MemoryProposalNotFoundError()

        if (
            pending.expires_at
            <= datetime.now(timezone.utc)
        ):
            self._pending.pop(proposal_id, None)
            raise MemoryProposalExpiredError()

        if pending.proposal.user_id != user_id:
            # Do not reveal another user's proposal.
            raise MemoryProposalNotFoundError()

        return pending

    def confirm(
        self,
        proposal_id: str,
        user_id: str,
    ) -> MemoryProposal:
        pending = self._get_owned_proposal(
            proposal_id=proposal_id,
            user_id=user_id,
        )

        self.command_service.confirm(
            pending.proposal
        )

        # A successful proposal can be used only once.
        self._pending.pop(proposal_id, None)

        return pending.proposal

    def reject(
        self,
        proposal_id: str,
        user_id: str,
    ) -> MemoryProposal:
        pending = self._get_owned_proposal(
            proposal_id=proposal_id,
            user_id=user_id,
        )

        self._pending.pop(proposal_id, None)

        return pending.proposal