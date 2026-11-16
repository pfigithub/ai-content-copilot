import uuid
from typing import Optional, TypedDict
from psycopg_pool import AsyncConnectionPool
from app.config import settings
from app.auth.security import hash_password

class UserRecord(TypedDict):
    id: str
    username: str
    password_hash: str
    role: str
    status: str

class AuthRepository:
    """Repository for managing users and roles in PostgreSQL."""
    
    def __init__(self, pool: AsyncConnectionPool):
        self.pool = pool

    async def setup(self) -> None:
        """Create the users table and inject the root admin if not exists."""
        async with self.pool.connection() as connection:
            # 1. Create table
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id UUID PRIMARY KEY,
                    username VARCHAR(64) UNIQUE NOT NULL,
                    password_hash VARCHAR(255) NOT NULL,
                    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'user')),
                    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'pending', 'suspended')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            
            # 2. Inject Root Admin from environment variables
            admin_username = settings.first_superuser_username
            admin_password_plain = settings.first_superuser_password
            
            # Check if admin already exists
            cursor = await connection.execute("SELECT id FROM users WHERE username = %s", (admin_username,))
            exists = await cursor.fetchone()
            
            if not exists:
                admin_id = str(uuid.uuid4())
                hashed_pw = hash_password(admin_password_plain)
                
                await connection.execute(
                    """
                    INSERT INTO users (id, username, password_hash, role, status)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (admin_id, admin_username, hashed_pw, "admin", "active")
                )
                print(f"✅ Root Admin '{admin_username}' successfully provisioned.")

    async def create_user(self, username: str, password_hash: str, role: str = "user", status: str = "pending") -> UserRecord:
        """Insert a new user into the database."""
        user_id = str(uuid.uuid4())
        
        async with self.pool.connection() as connection:
            try:
                await connection.execute(
                    """
                    INSERT INTO users (id, username, password_hash, role, status)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (user_id, username, password_hash, role, status)
                )
                return await self.get_user_by_username(username)
            except Exception as e:
                # E.g., UniqueViolation if username already exists
                raise ValueError(f"Could not create user: {e}")

    async def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        """Fetch a user by their exact username."""
        async with self.pool.connection() as connection:
            cursor = await connection.execute(
                "SELECT id, username, password_hash, role, status FROM users WHERE username = %s",
                (username,)
            )
            row = await cursor.fetchone()
            
            if row:
                return {
                    "id": str(row[0]),
                    "username": row[1],
                    "password_hash": row[2],
                    "role": row[3],
                    "status": row[4]
                }
            return None
            
    async def update_user_status(self, user_id: str, status: str) -> None:
        """Used by admins to approve or suspend accounts."""
        async with self.pool.connection() as connection:
            await connection.execute("UPDATE users SET status = %s WHERE id = %s", (status, user_id))