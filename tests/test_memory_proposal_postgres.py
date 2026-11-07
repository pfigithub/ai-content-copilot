
import asyncio
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from psycopg_pool import AsyncConnectionPool

from app.memory.commands import MemoryProposal
from app.memory.proposal_repository import (
    MemoryProposalNotFoundError,
    PendingMemoryProposal,
    PostgresMemoryProposalRepository,
)


@pytest.fixture
def anyio_backend():
    """Run this module with a Psycopg-compatible Windows event loop."""
    return (
        "asyncio",
        {"loop_factory": asyncio.SelectorEventLoop},
    )


@pytest.mark.anyio
async def test_postgres_serializes_concurrent_confirmations():
    """Only one worker can consume a proposal during concurrent confirmation."""
    postgres_uri = os.getenv("TEST_POSTGRES_URI")

    if not postgres_uri:
        pytest.skip("TEST_POSTGRES_URI is not configured.")

    pool = AsyncConnectionPool(
        conninfo=postgres_uri,
        min_size=1,
        max_size=3,
        open=False,
    )

    proposal_id = str(uuid4())
    user_id = f"postgres-test-{uuid4().hex}"
    proposal_created = False

    pending = PendingMemoryProposal(
        proposal_id=proposal_id,
        proposal=MemoryProposal(
            user_id=user_id,
            memory_id="concurrency-test-memory",
            content="This proposal must be consumed only once.",
            memory_type="preference",
        ),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )

    callback_calls = 0

    async def confirmation_callback(proposal: MemoryProposal) -> None:
        nonlocal callback_calls
        callback_calls += 1

        # Keep the row lock long enough for the competing request
        # to attempt confirmation while the first transaction is active.
        await asyncio.sleep(0.1)

    repository_a = PostgresMemoryProposalRepository(pool)
    repository_b = PostgresMemoryProposalRepository(pool)

    await pool.open()

    try:
        await pool.wait()
        await repository_a.setup()
        await repository_a.create(pending)
        proposal_created = True

        results = await asyncio.gather(
            repository_a.confirm(
                proposal_id=proposal_id,
                user_id=user_id,
                on_confirm=confirmation_callback,
            ),
            repository_b.confirm(
                proposal_id=proposal_id,
                user_id=user_id,
                on_confirm=confirmation_callback,
            ),
            return_exceptions=True,
        )

        successes = [
            result
            for result in results
            if isinstance(result, MemoryProposal)
        ]

        not_found_errors = [
            result
            for result in results
            if isinstance(result, MemoryProposalNotFoundError)
        ]

        assert len(successes) == 1
        assert len(not_found_errors) == 1
        assert callback_calls == 1

    finally:
        try:
            if proposal_created:
                async with pool.connection() as connection:
                    await connection.execute(
                        """
                        DELETE FROM memory_proposals
                        WHERE proposal_id = %s
                        """,
                        (proposal_id,),
                    )
        finally:
            await pool.close()
