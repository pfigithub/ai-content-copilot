import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Compiled once at import time; cheaper than recompiling per request.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


class UserRegisterRequest(BaseModel):
    """Schema for new user registration."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=64,
        description="Unique username (3-64 chars, alphanumeric and underscores).",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password (8-128 characters).",
    )

    @field_validator("username", mode="before")
    @classmethod
    def strip_username(cls, value):
        """Strip surrounding whitespace BEFORE length/pattern checks.

        Without this, `"  abc  "` would fail min_length=3 on the raw value
        even though the logical username is valid.
        """
        return value.strip() if isinstance(value, str) else value

    @field_validator("username")
    @classmethod
    def validate_username_charset(cls, value: str) -> str:
        """Reject anything outside [A-Za-z0-9_] with a clear message.

        The old implementation used `not value.isalnum() and "_" not in value`,
        which incorrectly accepted strings like "a!b_c" because they contain
        an underscore. A full-match regex is the only correct way.
        """
        if not _USERNAME_RE.fullmatch(value):
            raise ValueError(
                "Username can only contain letters, numbers, and underscores."
            )
        return value


class UserLoginRequest(BaseModel):
    """Schema for logging in. Bounded to prevent oversized-payload DoS."""

    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


class UserProfileResponse(BaseModel):
    """Public user profile. NEVER includes password_hash."""

    id: str
    username: str
    role: Literal["admin", "user"]
    status: Literal["active", "pending", "suspended"]


class TokenResponse(BaseModel):
    """JWT login response with the embedded public profile."""

    access_token: str
    token_type: str = "bearer"
    # Seconds until the access token expires. Default keeps old clients
    # working if they happen to omit the field while we migrate.
    expires_in: int = 900
    # Optional to remain backward-compatible with in-flight code paths.
    user: UserProfileResponse | None = None