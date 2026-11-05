
import hashlib
import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBearer,
)

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
    """Authenticate a Bearer API key and return its server-side user ID."""

    if not settings.api_key_hashes:
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
