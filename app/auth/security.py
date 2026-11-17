import logging
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

from app.config import settings


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Password hashing (Argon2id)                                        #
# ------------------------------------------------------------------ #

# OWASP-recommended Argon2id parameters (as of 2024). Locking them here
# means a future `argon2-cffi` upgrade will not silently change our
# security properties. If we ever raise these, `check_needs_rehash`
# becomes useful again.
_hasher = PasswordHasher(
    time_cost=3,
    memory_cost=65536,   # 64 MiB
    parallelism=4,
)

# Enforced at registration; kept in sync with the schema's min_length.
MIN_PASSWORD_LENGTH = 8


def hash_password(password: str) -> str:
    """Hash a plaintext password with Argon2id."""
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )
    return _hasher.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Constant-time verification of an Argon2id hash."""
    try:
        return _hasher.verify(hashed_password, plain_password)
    except VerifyMismatchError:
        return False


# ------------------------------------------------------------------ #
# JWT access tokens                                                   #
# ------------------------------------------------------------------ #

# 15 minutes. Short enough to limit blast radius if leaked, long enough
# that users aren't forced to re-login constantly. The next stage will
# add a refresh token so users never notice the rotation.
ACCESS_TOKEN_TTL_SECONDS = 15 * 60

# Reusable dummy hash used for login timing-attack mitigation.
# It never matches any password because the salt is arbitrary and the
# digest below is the hash of a random secret, discarded.
_DUMMY_HASH = _hasher.hash("__timing_attack_dummy__")


def create_access_token(user_id: str, role: str) -> str:
    """Issue a short-lived JWT. Signature is unchanged for compatibility."""
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured. Cannot issue tokens.")
    if role not in ("admin", "user"):
        raise ValueError(f"Invalid role: {role}")

    now = datetime.now(timezone.utc)
    payload = {
        "iss": settings.jwt_issuer,
        "aud": settings.jwt_audience,
        "iat": now,
        "exp": now + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS),
        "sub": user_id,
        "role": role,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def decode_access_token(token: str) -> dict:
    """Validate all claims and return the payload. Raises ValueError on failure."""
    if not settings.jwt_secret:
        raise RuntimeError("JWT_SECRET is not configured.")

    try:
        return jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"],  # pin the algorithm explicitly
            audience=settings.jwt_audience,
            issuer=settings.jwt_issuer,
        )
    except jwt.ExpiredSignatureError as exc:
        raise ValueError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc


def get_dummy_hash() -> str:
    """Return a pre-computed hash for timing-attack mitigation on login.

    The login endpoint runs `verify_password(input, get_dummy_hash())` when
    the username does not exist, so an attacker cannot distinguish an
    unknown username from a wrong password by response time.
    """
    return _DUMMY_HASH