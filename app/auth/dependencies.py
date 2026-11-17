import logging

from fastapi import Depends, HTTPException, Request, status

from app.auth.repository import AuthRepository, UserRecord
from app.dependencies import get_current_user


logger = logging.getLogger(__name__)


def get_auth_repository(request: Request) -> AuthRepository:
    """Retrieve the application-wide auth repository.

    Fails with 500 if the lifespan startup did not register it, which
    indicates a configuration bug that should never reach production.
    """
    repository = getattr(request.app.state, "auth_repository", None)
    if repository is None:
        logger.error("Auth repository is not initialised on app.state.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth repository is not initialised.",
        )
    return repository


async def get_current_user_record(
    user_id: str = Depends(get_current_user),
    repository: AuthRepository = Depends(get_auth_repository),
) -> UserRecord:
    """Return the authenticated user's full record after status checks.

    Adds two guarantees the lightweight `get_current_user` cannot provide:

    - The user still exists in the database. A token stays valid for at
      most 15 minutes after deletion, but we reject immediately.
    - The account is still active. Suspending a user therefore takes
      effect on the very next request, not after the token expires.
    """
    user = await repository.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account no longer exists.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user['status']}.",
        )

    return user


async def require_admin(
    user: UserRecord = Depends(get_current_user_record),
) -> UserRecord:
    """Restrict an endpoint to administrators."""
    if user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required.",
        )
    return user