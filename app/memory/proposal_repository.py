
import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Protocol

from psycopg_pool import AsyncConnectionPool

from app.memory.commands import MemoryProposal, MemoryType


class MemoryProposalNotFoundError(Exception):
    """Raised when a proposal does not exist or is not owned by the caller."""


class MemoryProposalExpiredError(Exception):
    """Raised when a proposal is no longer valid."""


@dataclass(frozen=True)
class PendingMemoryProposal:
    """Represent a proposal awaiting explicit user confirmation."""

    proposal_id: str
    proposal: MemoryProposal
    expires_at: datetime


ConfirmCallback = Callable[[MemoryProposal], Awaitable[None]]


class MemoryProposalRepository(Protocol):
    """Define persistent storage operations for pending memory proposals."""

    async def create(self, pending: PendingMemoryProposal) -> None:
        """Persist a new pending proposal."""

    async def confirm(
        self,
        proposal_id: str,
        user_id: str,
        on_confirm: ConfirmCallback,
    ) -> MemoryProposal:
        """Atomically consume a proposal after its confirmation callback succeeds."""

    async def reject(
        self,
        proposal_id: str,
        user_id: str,
    ) -> MemoryProposal:
        """Consume a proposal without writing it to long-term memory."""


class InMemoryMemoryProposalRepository:
    """Store pending proposals in memory for tests and temporary execution."""

    def __init__(self) -> None:
        self._pending: dict[str, PendingMemoryProposal] = {}
        self._lock = asyncio.Lock()

    async def create(self, pending: PendingMemoryProposal) -> None:
        """Add a proposal and discard expired entries."""
        now = datetime.now(timezone.utc)

        async with self._lock:
            expired_ids = [
                proposal_id
                for proposal_id, item in self._pending.items()
                if item.expires_at <= now
            ]

            for proposal_id in expired_ids:
                self._pending.pop(proposal_id, None)

            self._pending[pending.proposal_id] = pending

    def _get_owned_locked(
        self,
        proposal_id: str,
        user_id: str,
    ) -> PendingMemoryProposal:
        """Retrieve an unexpired proposal while the repository lock is held."""
        pending = self._pending.get(proposal_id)

        if pending is None or pending.proposal.user_id != user_id:
            raise MemoryProposalNotFoundError()

        if pending.expires_at <= datetime.now(timezone.utc):
            self._pending.pop(proposal_id, None)
            raise MemoryProposalExpiredError()

        return pending

    async def confirm(
        self,
        proposal_id: str,
        user_id: str,
        on_confirm: ConfirmCallback,
    ) -> MemoryProposal:
        """Write the proposal before removing it from the in-memory repository."""
        async with self._lock:
            pending = self._get_owned_locked(proposal_id, user_id)

            await on_confirm(pending.proposal)

            self._pending.pop(proposal_id, None)
            return pending.proposal

    async def reject(
        self,
        proposal_id: str,
        user_id: str,
    ) -> MemoryProposal:
        """Remove an owned proposal without saving it."""
        async with self._lock:
            pending = self._get_owned_locked(proposal_id, user_id)
            self._pending.pop(proposal_id, None)
            return pending.proposal


class PostgresMemoryProposalRepository:
    """Persist proposals in PostgreSQL and serialize concurrent decisions."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def setup(self) -> None:
        """Create the proposal table and its lookup index when absent."""
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_proposals (
                    proposal_id UUID PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL
                        CHECK (
                            memory_type IN (
                                'preference',
                                'self_reported_fact'
                            )
                        ),
                    confirmation_required BOOLEAN NOT NULL DEFAULT TRUE,
                    expires_at TIMESTAMPTZ NOT NULL
                )
                """
            )

            await connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_memory_proposals_user_expires
                ON memory_proposals (user_id, expires_at)
                """
            )

    async def create(self, pending: PendingMemoryProposal) -> None:
        """Insert a proposal and clean up expired records."""
        proposal = pending.proposal

        async with self.pool.connection() as connection:
            await connection.execute(
                """
                DELETE FROM memory_proposals
                WHERE expires_at <= %s
                """,
                (datetime.now(timezone.utc),),
            )

            await connection.execute(
                """
                INSERT INTO memory_proposals (
                    proposal_id,
                    user_id,
                    memory_id,
                    content,
                    memory_type,
                    confirmation_required,
                    expires_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    pending.proposal_id,
                    proposal.user_id,
                    proposal.memory_id,
                    proposal.content,
                    proposal.memory_type,
                    proposal.confirmation_required,
                    pending.expires_at,
                ),
            )

    @staticmethod
    def _to_pending(row: tuple) -> PendingMemoryProposal:
        """Convert a PostgreSQL row into the application's proposal model."""
        (
            proposal_id,
            user_id,
            memory_id,
            content,
            memory_type,
            confirmation_required,
            expires_at,
        ) = row

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        proposal = MemoryProposal(
            user_id=user_id,
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            confirmation_required=confirmation_required,
        )

        return PendingMemoryProposal(
            proposal_id=str(proposal_id),
            proposal=proposal,
            expires_at=expires_at,
        )

    async def _select_for_update(
        self,
        connection,
        proposal_id: str,
        user_id: str,
    ) -> PendingMemoryProposal:
        """Load an owned proposal and lock its row until transaction completion."""
        cursor = await connection.execute(
            """
            SELECT
                proposal_id,
                user_id,
                memory_id,
                content,
                memory_type,
                confirmation_required,
                expires_at
            FROM memory_proposals
            WHERE proposal_id = %s
              AND user_id = %s
            FOR UPDATE
            """,
            (proposal_id, user_id),
        )

        row = await cursor.fetchone()

        if row is None:
            raise MemoryProposalNotFoundError()

        return self._to_pending(row)

    async def confirm(
        self,
        proposal_id: str,
        user_id: str,
        on_confirm: ConfirmCallback,
    ) -> MemoryProposal:
        """Lock, write, and consume a proposal in a serialized operation."""
        expired = False
        confirmed_proposal: MemoryProposal | None = None

        async with self.pool.connection() as connection:
            async with connection.transaction():
                pending = await self._select_for_update(
                    connection=connection,
                    proposal_id=proposal_id,
                    user_id=user_id,
                )

                if pending.expires_at <= datetime.now(timezone.utc):
                    await connection.execute(
                        """
                        DELETE FROM memory_proposals
                        WHERE proposal_id = %s
                        """,
                        (proposal_id,),
                    )
                    expired = True
                else:
                    # The row lock prevents another worker from confirming
                    # or rejecting this proposal until this transaction ends.
                    await on_confirm(pending.proposal)

                    await connection.execute(
                        """
                        DELETE FROM memory_proposals
                        WHERE proposal_id = %s
                        """,
                        (proposal_id,),
                    )

                    confirmed_proposal = pending.proposal

        if expired:
            raise MemoryProposalExpiredError()

        if confirmed_proposal is None:
            raise RuntimeError("The proposal confirmation did not complete.")

        return confirmed_proposal

    async def reject(
        self,
        proposal_id: str,
        user_id: str,
    ) -> MemoryProposal:
        """Lock and consume a proposal without writing a memory record."""
        expired = False
        rejected_proposal: MemoryProposal | None = None

        async with self.pool.connection() as connection:
            async with connection.transaction():
                pending = await self._select_for_update(
                    connection=connection,
                    proposal_id=proposal_id,
                    user_id=user_id,
                )

                if pending.expires_at <= datetime.now(timezone.utc):
                    await connection.execute(
                        """
                        DELETE FROM memory_proposals
                        WHERE proposal_id = %s
                        """,
                        (proposal_id,),
                    )
                    expired = True
                else:
                    await connection.execute(
                        """
                        DELETE FROM memory_proposals
                        WHERE proposal_id = %s
                        """,
                        (proposal_id,),
                    )
                    rejected_proposal = pending.proposal

        if expired:
            raise MemoryProposalExpiredError()

        if rejected_proposal is None:
            raise RuntimeError("The proposal rejection did not complete.")

        return rejected_proposal
