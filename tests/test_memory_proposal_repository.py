
import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from langgraph.store.memory import InMemoryStore

from app.memory.commands import MemoryCommandService, MemoryProposal
from app.memory.long_term_memory import UserMemoryStore
from app.memory.proposal_repository import (
    InMemoryMemoryProposalRepository,
    MemoryProposalExpiredError,
    MemoryProposalNotFoundError,
    PendingMemoryProposal,
)
from app.memory.proposal_service import MemoryProposalService
from app.memory.write_policy import MemoryWritePolicy


def build_service(
    repository: InMemoryMemoryProposalRepository,
    store: InMemoryStore,
) -> MemoryProposalService:
    """Build a new service instance around shared repository and memory storage."""
    user_memory_store = UserMemoryStore(store=store)
    write_policy = MemoryWritePolicy(memory_store=user_memory_store)
    command_service = MemoryCommandService(write_policy=write_policy)

    return MemoryProposalService(
        command_service=command_service,
        repository=repository,
    )


def test_proposal_survives_service_reconstruction():
    """A new service instance can access proposals in the shared repository."""
    repository = InMemoryMemoryProposalRepository()
    store = InMemoryStore()

    first_service = build_service(repository, store)

    pending = asyncio.run(
        first_service.propose(
            user_id="proposal-user-a",
            memory_id="language",
            content="The user prefers Persian responses.",
        )
    )

    # Rebuild the service to simulate an application worker restart.
    second_service = build_service(repository, store)

    confirmed = asyncio.run(
        second_service.confirm(
            proposal_id=pending.proposal_id,
            user_id="proposal-user-a",
        )
    )

    assert confirmed.memory_id == "language"

    saved_memory = asyncio.run(
        UserMemoryStore(store=store).get(
            user_id="proposal-user-a",
            memory_id="language",
        )
    )

    assert saved_memory is not None
    assert saved_memory["content"] == "The user prefers Persian responses."


def test_another_user_cannot_consume_proposal():
    """A failed cross-user attempt does not remove the owner's proposal."""
    repository = InMemoryMemoryProposalRepository()
    store = InMemoryStore()

    owner_service = build_service(repository, store)
    other_service = build_service(repository, store)

    pending = asyncio.run(
        owner_service.propose(
            user_id="proposal-owner",
            memory_id="private-memory",
            content="Private user information.",
        )
    )

    with pytest.raises(MemoryProposalNotFoundError):
        asyncio.run(
            other_service.confirm(
                proposal_id=pending.proposal_id,
                user_id="different-user",
            )
        )

    confirmed = asyncio.run(
        owner_service.confirm(
            proposal_id=pending.proposal_id,
            user_id="proposal-owner",
        )
    )

    assert confirmed.user_id == "proposal-owner"


def test_confirmed_proposal_cannot_be_consumed_twice():
    """A successfully confirmed proposal is removed from the repository."""
    repository = InMemoryMemoryProposalRepository()
    store = InMemoryStore()
    service = build_service(repository, store)

    pending = asyncio.run(
        service.propose(
            user_id="single-use-user",
            memory_id="single-use-memory",
            content="A memory that must be confirmed only once.",
        )
    )

    asyncio.run(
        service.confirm(
            proposal_id=pending.proposal_id,
            user_id="single-use-user",
        )
    )

    with pytest.raises(MemoryProposalNotFoundError):
        asyncio.run(
            service.confirm(
                proposal_id=pending.proposal_id,
                user_id="single-use-user",
            )
        )


def test_rejected_proposal_is_not_written():
    """Rejecting a proposal consumes it without saving long-term memory."""
    repository = InMemoryMemoryProposalRepository()
    store = InMemoryStore()
    service = build_service(repository, store)

    pending = asyncio.run(
        service.propose(
            user_id="reject-user",
            memory_id="rejected-memory",
            content="This should never be saved.",
        )
    )

    rejected = asyncio.run(
        service.reject(
            proposal_id=pending.proposal_id,
            user_id="reject-user",
        )
    )

    assert rejected.memory_id == "rejected-memory"

    memory = asyncio.run(
        UserMemoryStore(store=store).get(
            user_id="reject-user",
            memory_id="rejected-memory",
        )
    )

    assert memory is None


def test_expired_proposal_is_removed_and_rejected():
    """An expired proposal cannot be confirmed."""
    repository = InMemoryMemoryProposalRepository()

    pending = PendingMemoryProposal(
        proposal_id="expired-proposal",
        proposal=MemoryProposal(
            user_id="expired-user",
            memory_id="expired-memory",
            content="Expired content.",
            memory_type="preference",
        ),
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    asyncio.run(repository.create(pending))

    async def confirmation_callback(proposal: MemoryProposal) -> None:
        raise AssertionError("Expired proposals must never be written.")

    with pytest.raises(MemoryProposalExpiredError):
        asyncio.run(
            repository.confirm(
                proposal_id=pending.proposal_id,
                user_id="expired-user",
                on_confirm=confirmation_callback,
            )
        )

    with pytest.raises(MemoryProposalNotFoundError):
        asyncio.run(
            repository.reject(
                proposal_id=pending.proposal_id,
                user_id="expired-user",
            )
        )
