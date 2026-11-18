"""Refresh token persistence.

One row per issued refresh token. Two design choices are critical:

1. **Rotation** — every refresh replaces the old token with a new one.
   If a rotated token is ever presented again, the whole *family* of
   tokens is revoked. This is the OAuth 2.1 standard response to a
   suspected token theft.

2. **Family** — tokens issued from the same login belong to the same
   family. Rotation keeps the family id so that revoking "everything
   issued to this device" is one UPDATE. Revoking "everything for this
   user" iterates families but stays cheap.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from psycopg_pool import AsyncConnectionPool


logger = logging.getLogger(__name__)


class RotationStatus(str, Enum):
    OK = "ok"
    NOT_FOUND = "not_found"
    EXPIRED = "expired"
    REVOKED = "revoked"
    REUSE_DETECTED = "reuse_detected"


@dataclass(frozen=True)
class RefreshSessionRecord:
    id: str
    user_id: str
    family_id: str
    token_hash: str
    expires_at: datetime
    rotated_at: Optional[datetime]
    revoked: bool


@dataclass
class RotationResult:
    status: RotationStatus
    new_record: Optional[RefreshSessionRecord] = None


class PostgresRefreshTokenRepository:
    """Persist refresh sessions in PostgreSQL with atomic rotation."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def setup(self) -> None:
        """Create the table and indexes if absent."""
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS refresh_sessions (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL
                        REFERENCES users(id) ON DELETE CASCADE,
                    family_id UUID NOT NULL,
                    token_hash TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMPTZ NOT NULL,
                    rotated_at TIMESTAMPTZ,
                    revoked BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_refresh_user
                ON refresh_sessions (user_id)
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_refresh_family
                ON refresh_sessions (family_id)
                """
            )

    async def insert_session(
        self,
        user_id: str,
        token_hash: str,
        expires_at: datetime,
        family_id: Optional[str] = None,
    ) -> RefreshSessionRecord:
        """Insert a session. If family_id is None, a new family starts."""
        session_id = str(uuid.uuid4())
        actual_family = family_id or str(uuid.uuid4())
        async with self.pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO refresh_sessions
                    (id, user_id, family_id, token_hash, expires_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (session_id, user_id, actual_family, token_hash, expires_at),
            )
        return RefreshSessionRecord(
            id=session_id,
            user_id=user_id,
            family_id=actual_family,
            token_hash=token_hash,
            expires_at=expires_at,
            rotated_at=None,
            revoked=False,
        )

    async def rotate(
        self,
        old_hash: str,
        new_hash: str,
        new_expires_at: datetime,
    ) -> RotationResult:
        """Atomically rotate a refresh token.

        If the old token was already rotated (reuse) or revoked, the
        entire family is revoked and REUSE_DETECTED is returned.
        """
        async with self.pool.connection() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    """
                    SELECT id, user_id, family_id, expires_at,
                           rotated_at, revoked
                    FROM refresh_sessions
                    WHERE token_hash = %s
                    FOR UPDATE
                    """,
                    (old_hash,),
                )
                row = await cur.fetchone()
                if row is None:
                    return RotationResult(RotationStatus.NOT_FOUND)

                _, user_id, family_id, expires_at, rotated_at, revoked = row

                if revoked:
                    return RotationResult(RotationStatus.REVOKED)

                if expires_at <= datetime.now(timezone.utc):
                    return RotationResult(RotationStatus.EXPIRED)

                if rotated_at is not None:
                    # REUSE! Revoke the whole family so both the attacker
                    # and the legitimate user are forced to log in again.
                    await conn.execute(
                        """
                        UPDATE refresh_sessions SET revoked = TRUE
                        WHERE family_id = %s
                        """,
                        (family_id,),
                    )
                    logger.warning(
                        "Refresh token reuse detected for user %s; "
                        "family %s revoked.", user_id, family_id,
                    )
                    return RotationResult(RotationStatus.REUSE_DETECTED)

                await conn.execute(
                    """
                    UPDATE refresh_sessions SET rotated_at = NOW()
                    WHERE token_hash = %s
                    """,
                    (old_hash,),
                )
                new_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT INTO refresh_sessions
                        (id, user_id, family_id, token_hash, expires_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (new_id, user_id, family_id, new_hash, new_expires_at),
                )

        return RotationResult(
            RotationStatus.OK,
            RefreshSessionRecord(
                id=new_id,
                user_id=user_id,
                family_id=family_id,
                token_hash=new_hash,
                expires_at=new_expires_at,
                rotated_at=None,
                revoked=False,
            ),
        )

    async def revoke_session(self, token_hash: str) -> bool:
        """Revoke a single refresh token. Used by logout."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE refresh_sessions SET revoked = TRUE
                WHERE token_hash = %s AND revoked = FALSE
                """,
                (token_hash,),
            )
            return cur.rowcount > 0

    async def revoke_all_for_user(self, user_id: str) -> int:
        """Revoke every refresh token for a user. Used on password change."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                UPDATE refresh_sessions SET revoked = TRUE
                WHERE user_id = %s AND revoked = FALSE
                """,
                (user_id,),
            )
            return cur.rowcount

    async def cleanup_expired(self, retain_days: int = 30) -> int:
        """Delete sessions expired more than retain_days ago."""
        async with self.pool.connection() as conn:
            cur = await conn.execute(
                """
                DELETE FROM refresh_sessions
                WHERE expires_at < NOW() - (%s * INTERVAL '1 day')
                """,
                (retain_days,),
            )
            return cur.rowcount


class InMemoryRefreshTokenRepository:
    """In-memory equivalent for tests and APP_ENV=test."""

    def __init__(self) -> None:
        self._rows: dict[str, RefreshSessionRecord] = {}
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
        return None

    async def insert_session(
        self,
        user_id: str,
        token_hash: str,
        expires_at: datetime,
        family_id: Optional[str] = None,
    ) -> RefreshSessionRecord:
        async with self._lock:
            record = RefreshSessionRecord(
                id=str(uuid.uuid4()),
                user_id=user_id,
                family_id=family_id or str(uuid.uuid4()),
                token_hash=token_hash,
                expires_at=expires_at,
                rotated_at=None,
                revoked=False,
            )
            self._rows[token_hash] = record
            return record

    async def rotate(
        self,
        old_hash: str,
        new_hash: str,
        new_expires_at: datetime,
    ) -> RotationResult:
        async with self._lock:
            old = self._rows.get(old_hash)
            if old is None:
                return RotationResult(RotationStatus.NOT_FOUND)
            if old.revoked:
                return RotationResult(RotationStatus.REVOKED)
            if old.expires_at <= datetime.now(timezone.utc):
                return RotationResult(RotationStatus.EXPIRED)
            if old.rotated_at is not None:
                # Reuse detected: revoke the family.
                for h, r in list(self._rows.items()):
                    if r.family_id == old.family_id and not r.revoked:
                        self._rows[h] = RefreshSessionRecord(
                            **{**r.__dict__, "revoked": True}
                        )
                return RotationResult(RotationStatus.REUSE_DETECTED)

            self._rows[old_hash] = RefreshSessionRecord(
                **{**old.__dict__,
                   "rotated_at": datetime.now(timezone.utc)}
            )
            new = RefreshSessionRecord(
                id=str(uuid.uuid4()),
                user_id=old.user_id,
                family_id=old.family_id,
                token_hash=new_hash,
                expires_at=new_expires_at,
                rotated_at=None,
                revoked=False,
            )
            self._rows[new_hash] = new
            return RotationResult(RotationStatus.OK, new)

    async def revoke_session(self, token_hash: str) -> bool:
        async with self._lock:
            row = self._rows.get(token_hash)
            if row is None or row.revoked:
                return False
            self._rows[token_hash] = RefreshSessionRecord(
                **{**row.__dict__, "revoked": True}
            )
            return True

    async def revoke_all_for_user(self, user_id: str) -> int:
        async with self._lock:
            count = 0
            for h, r in list(self._rows.items()):
                if r.user_id == user_id and not r.revoked:
                    self._rows[h] = RefreshSessionRecord(
                        **{**r.__dict__, "revoked": True}
                    )
                    count += 1
            return count

    async def cleanup_expired(self, retain_days: int = 30) -> int:
        # In-memory store is ephemeral; nothing to clean.
        return 0