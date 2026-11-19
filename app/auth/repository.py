import logging
import uuid
from typing import Optional, TypedDict

from psycopg_pool import AsyncConnectionPool

from app.auth.security import hash_password
from app.config import settings


logger = logging.getLogger(__name__)


VALID_ROLES = {"admin", "user"}
VALID_STATUSES = {"active", "pending", "suspended"}


class UserRecord(TypedDict):
    id: str
    username: str
    password_hash: str
    role: str
    status: str


class AuthRepository:
    """Persistent user storage backed by PostgreSQL."""

    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    # -------------------------------------------------------------- #
    # Schema + bootstrap                                              #
    # -------------------------------------------------------------- #

    async def setup(self) -> None:
        """Create the users table and provision the root admin if absent."""
        async with self.pool.connection() as connection:
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    username VARCHAR(64) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'user')),
                    status VARCHAR(20) NOT NULL
                        CHECK (status IN ('active', 'pending', 'suspended')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await self._provision_root_admin(connection)

    async def _provision_root_admin(self, connection) -> None:
        """Idempotent, race-safe admin bootstrap.

        - Validates credentials BEFORE hashing, so a missing password
          fails with a clear message instead of a cryptic hash error.
        - Uses `ON CONFLICT DO NOTHING` so multiple uvicorn workers
          racing to provision can't crash one another.
        """
        username = (settings.first_superuser_username or "").strip()
        password = settings.first_superuser_password or ""

        if not username:
            raise RuntimeError(
                "FIRST_SUPERUSER_USERNAME is empty. "
                "Set it in .env before starting the application."
            )
        if len(password) < 8:
            raise RuntimeError(
                "FIRST_SUPERUSER_PASSWORD must be at least 8 characters."
            )

        await connection.execute(
            """
            INSERT INTO users (id, username, password_hash, role, status)
            VALUES (%s, %s, %s, 'admin', 'active')
            ON CONFLICT (username) DO NOTHING
            """,
            (str(uuid.uuid4()), username, hash_password(password)),
        )
        # Log only the username; never log the password.
        logger.info("Root admin '%s' ensured.", username)

    # -------------------------------------------------------------- #
    # Reads                                                           #
    # -------------------------------------------------------------- #

    async def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT id, username, password_hash, role, status
                FROM users WHERE username = %s
                """,
                (username,),
            )
            row = await cursor.fetchone()
            return self._row_to_record(row) if row else None

    async def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                """
                SELECT id, username, password_hash, role, status
                FROM users WHERE id = %s
                """,
                (user_id,),
            )
            row = await cursor.fetchone()
            return self._row_to_record(row) if row else None

    async def list_users(self, status: Optional[str] = None) -> list[UserRecord]:
        """Return all users, optionally filtered by status. Admin-only."""
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"Invalid status filter: {status}")

        async with self.pool.connection() as connection:
            if status is None:
                cursor = await connection.execute(
                    """
                    SELECT id, username, password_hash, role, status
                    FROM users ORDER BY created_at
                    """
                )
            else:
                cursor = await connection.execute(
                    """
                    SELECT id, username, password_hash, role, status
                    FROM users WHERE status = %s ORDER BY created_at
                    """,
                    (status,),
                )
            rows = await cursor.fetchall()
            return [self._row_to_record(row) for row in rows]

    # -------------------------------------------------------------- #
    # Writes                                                          #
    # -------------------------------------------------------------- #

    async def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "user",
        status: str = "pending",
    ) -> UserRecord:
        """Insert a new user. Raises ValueError on invalid input or duplicate."""
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        user_id = str(uuid.uuid4())
        async with self.pool.connection() as connection:
            try:
                await connection.execute(
                    """
                    INSERT INTO users (id, username, password_hash, role, status)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, username, password_hash, role, status),
                )
            except Exception as exc:
                # psycopg raises UniqueViolation on duplicate username;
                # surface it as ValueError so callers can map to 409.
                raise ValueError(f"Could not create user: {exc}") from exc

        record = await self.get_user_by_username(username)
        if record is None:
            raise RuntimeError("User vanished immediately after creation.")
        return record
    
    async def update_password(
        self,
        user_id: str,
        new_password_hash: str,
    ) -> bool:
        """Replace a user's password hash. Returns True if a row was updated.

        We deliberately do NOT check the old hash here: that responsibility
        belongs to the endpoint (which has access to the plaintext input).
        The repository only knows hashes.
        """
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (new_password_hash, user_id),
            )
            return cursor.rowcount > 0

    async def update_user_status(self, user_id: str, status: str) -> bool:
        """Admin action: set a user's status. Returns True if a row was updated."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "UPDATE users SET status = %s WHERE id = %s",
                (status, user_id),
            )
            return cursor.rowcount > 0

    async def delete_user(self, user_id: str) -> bool:
        """Admin action: permanently remove a user. Returns True if deleted."""
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "DELETE FROM users WHERE id = %s",
                (user_id,),
            )
            return cursor.rowcount > 0

    # -------------------------------------------------------------- #
    # Helpers                                                         #
    # -------------------------------------------------------------- #

    @staticmethod
    def _row_to_record(row: tuple) -> UserRecord:
        return {
            "id": str(row[0]),
            "username": row[1],
            "password_hash": row[2],
            "role": row[3],
            "status": row[4],
        }