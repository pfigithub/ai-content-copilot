"""Tests for refresh token rotation and reuse detection."""

from datetime import datetime, timedelta, timezone

import pytest

from app.auth.refresh_repository import (
    InMemoryRefreshTokenRepository,
    RotationStatus,
)


@pytest.fixture
def repo() -> InMemoryRefreshTokenRepository:
    return InMemoryRefreshTokenRepository()


@pytest.mark.asyncio
async def test_insert_and_rotate_happy_path(repo):
    future = datetime.now(timezone.utc) + timedelta(days=7)
    old = await repo.insert_session("user-1", "hash-old", future)

    new_expires = datetime.now(timezone.utc) + timedelta(days=7)
    result = await repo.rotate("hash-old", "hash-new", new_expires)

    assert result.status == RotationStatus.OK
    assert result.new_record is not None
    assert result.new_record.family_id == old.family_id
    assert result.new_record.token_hash == "hash-new"


@pytest.mark.asyncio
async def test_rotate_unknown_token(repo):
    result = await repo.rotate("unknown", "new", datetime.now(timezone.utc))
    assert result.status == RotationStatus.NOT_FOUND


@pytest.mark.asyncio
async def test_rotate_expired_token(repo):
    past = datetime.now(timezone.utc) - timedelta(days=1)
    await repo.insert_session("user-1", "hash-old", past)
    result = await repo.rotate(
        "hash-old", "hash-new", datetime.now(timezone.utc) + timedelta(days=7)
    )
    assert result.status == RotationStatus.EXPIRED


@pytest.mark.asyncio
async def test_reuse_detection_revokes_family(repo):
    future = datetime.now(timezone.utc) + timedelta(days=7)
    await repo.insert_session("user-1", "hash-1", future)

    # Rotate once: hash-1 -> hash-2.
    r1 = await repo.rotate("hash-1", "hash-2", future)
    assert r1.status == RotationStatus.OK

    # Try to rotate hash-1 again (reuse).
    r2 = await repo.rotate("hash-1", "hash-3", future)
    assert r2.status == RotationStatus.REUSE_DETECTED

    # hash-2 must be revoked too (family-wide revocation).
    r3 = await repo.rotate("hash-2", "hash-4", future)
    assert r3.status == RotationStatus.REVOKED


@pytest.mark.asyncio
async def test_revoke_all_for_user(repo):
    future = datetime.now(timezone.utc) + timedelta(days=7)
    await repo.insert_session("user-1", "hash-a", future)
    await repo.insert_session("user-1", "hash-b", future)
    await repo.insert_session("user-2", "hash-c", future)

    count = await repo.revoke_all_for_user("user-1")
    assert count == 2

    # user-1 tokens are revoked.
    r = await repo.rotate("hash-a", "x", future)
    assert r.status == RotationStatus.REVOKED

    # user-2 token is untouched.
    r = await repo.rotate("hash-c", "y", future)
    assert r.status == RotationStatus.OK