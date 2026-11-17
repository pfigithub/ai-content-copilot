import hashlib
import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

from app.auth.security import decode_access_token
from app.config import settings
from app.providers.ai_provider import AIProvider
from app.providers.ollama_provider import OllamaProvider


_bearer_scheme = HTTPBearer(auto_error=False)


def get_ai_provider() -> AIProvider:
    """Return the default local AI provider."""

    return OllamaProvider()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(
        _bearer_scheme
    ),
) -> str:
    """Authenticate a Bearer token and return its server-side user ID.

    Two authentication methods are supported, tried in order:

    1. JWT access token issued by `/api/v1/auth/login` (the new default).
    2. Legacy SHA-256 API key configured via `API_KEY_HASHES` (kept so
       existing integrations and integration tests continue to work
       while clients migrate).

    The return type must remain a plain string (the user_id) because all
    existing endpoints treat the dependency's output as an opaque user
    identifier. Anything else would break every protected route.
    """

    if not settings.api_key_hashes and not settings.jwt_secret:
        # No authentication method is configured at all — fail closed.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured.",
        )

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    if not token or token != token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # ---- Primary path: JWT -------------------------------------- #
    if settings.jwt_secret:
        try:
            payload = decode_access_token(token)
            return payload["sub"]
        except (ValueError, RuntimeError):
            # Not a valid JWT (malformed, expired, wrong audience, ...).
            # Fall through to the legacy API-key check below.
            pass

    # ---- Legacy path: API key hashes ---------------------------- #
    if settings.api_key_hashes:
        candidate_digest = hashlib.sha256(
            token.encode("utf-8")
        ).hexdigest()

        for expected_digest, user_id in settings.api_key_hashes.items():
            if hmac.compare_digest(candidate_digest, expected_digest):
                return user_id

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )