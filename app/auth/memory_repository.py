import asyncio
import logging
import uuid
from typing import Optional

from app.auth.repository import UserRecord, VALID_ROLES, VALID_STATUSES
from app.auth.security import hash_password
from app.config import settings


logger = logging.getLogger(__name__)


class InMemoryAuthRepository:
    """In-memory replacement for tests and offline local runs.

    Same interface as `AuthRepository` so callers are agnostic. Guarded by
    an asyncio lock because `setup` and `create_user` can race when tests
    run concurrently.
    """

    def __init__(self) -> None:
        # Keyed by user_id.
        self._users: dict[str, UserRecord] = {}
        self._lock = asyncio.Lock()

    async def setup(self) -> None:
        """Provision the root admin if credentials are configured.

        Unlike PostgreSQL mode, missing credentials are tolerated here so
        tests that don't care about the admin user aren't forced to set
        FIRST_SUPERUSER_* env vars.
        """
        username = (settings.first_superuser_username or "").strip()
        password = settings.first_superuser_password or ""
        if not username or len(password) < 8:
            logger.info(
                "In-memory auth: skipping admin provisioning "
                "(FIRST_SUPERUSER_* not configured)."
            )
            return

        async with self._lock:
            for user in self._users.values():
                if user["username"] == username:
                    return
            user_id = str(uuid.uuid4())
            self._users[user_id] = {
                "id": user_id,
                "username": username,
                "password_hash": hash_password(password),
                "role": "admin",
                "status": "active",
            }

    async def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        for user in self._users.values():
            if user["username"] == username:
                return dict(user)
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[UserRecord]:
        user = self._users.get(user_id)
        return dict(user) if user else None

    async def list_users(self, status: Optional[str] = None) -> list[UserRecord]:
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(f"Invalid status filter: {status}")
        return [
            dict(u) for u in self._users.values()
            if status is None or u["status"] == status
        ]

    async def create_user(
        self,
        username: str,
        password_hash: str,
        role: str = "user",
        status: str = "pending",
    ) -> UserRecord:
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")

        async with self._lock:
            if any(u["username"] == username for u in self._users.values()):
                raise ValueError(f"Username '{username}' is already taken.")
            user_id = str(uuid.uuid4())
            self._users[user_id] = {
                "id": user_id,
                "username": username,
                "password_hash": password_hash,
                "role": role,
                "status": status,
            }
            return dict(self._users[user_id])

    async def update_user_status(self, user_id: str, status: str) -> bool:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        async with self._lock:
            if user_id not in self._users:
                return False
            self._users[user_id]["status"] = status
            return True

    async def delete_user(self, user_id: str) -> bool:
        async with self._lock:
            return self._users.pop(user_id, None) is not None