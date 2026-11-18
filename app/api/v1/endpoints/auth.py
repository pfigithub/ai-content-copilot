import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import (
    get_auth_repository,
    get_current_user_record,
    get_refresh_token_repository,
    require_admin,
)
from app.auth.refresh_repository import RotationStatus
from app.auth.repository import AuthRepository, UserRecord
from app.auth.security import (
    ACCESS_TOKEN_TTL_SECONDS,
    REFRESH_TOKEN_TTL_SECONDS,
    create_access_token,
    create_refresh_token,
    get_dummy_hash,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.models.auth_schemas import (
    RefreshRequest,
    TokenResponse,
    UserLoginRequest,
    UserProfileResponse,
    UserRegisterRequest,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


def _to_profile(user: UserRecord) -> UserProfileResponse:
    """Convert an internal record into a public profile."""
    return UserProfileResponse(
        id=user["id"],
        username=user["username"],
        role=user["role"],
        status=user["status"],
    )


async def _issue_token_pair(
    user: UserRecord,
    refresh_repo,
    family_id: str | None = None,
) -> TokenResponse:
    """Create and persist a fresh access + refresh token pair.

    Shared by /login and /refresh. The refresh session is persisted
    BEFORE the access token is returned, so a database failure cannot
    leave a client holding an access token that has no matching
    refresh session.
    """
    raw_refresh = create_refresh_token()
    refresh_hash = hash_refresh_token(raw_refresh)
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=REFRESH_TOKEN_TTL_SECONDS
    )

    await refresh_repo.insert_session(
        user_id=user["id"],
        token_hash=refresh_hash,
        expires_at=expires_at,
        family_id=family_id,
    )

    access_token = create_access_token(
        user_id=user["id"],
        role=user["role"],
    )

    return TokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        expires_in=ACCESS_TOKEN_TTL_SECONDS,
        user=_to_profile(user),
    )


# ---------------------------------------------------------------- #
# Public endpoints                                                 #
# ---------------------------------------------------------------- #


@router.post(
    "/register",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register_user(
    request: UserRegisterRequest,
    repo: AuthRepository = Depends(get_auth_repository),
) -> UserProfileResponse:
    """Register a new user in 'pending' state awaiting admin approval."""
    existing = await repo.get_user_by_username(request.username)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken.",
        )

    hashed = hash_password(request.password)

    try:
        new_user = await repo.create_user(
            username=request.username,
            password_hash=hashed,
            role="user",
            status="pending",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken.",
        ) from exc

    return _to_profile(new_user)


@router.post("/login", response_model=TokenResponse)
async def login_user(
    request: UserLoginRequest,
    repo: AuthRepository = Depends(get_auth_repository),
    refresh_repo=Depends(get_refresh_token_repository),
) -> TokenResponse:
    """Authenticate and issue an access + refresh token pair."""
    user = await repo.get_user_by_username(request.username)

    hash_to_check = user["password_hash"] if user else get_dummy_hash()
    password_ok = verify_password(request.password, hash_to_check)

    if not user or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    if user["status"] == "pending":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is pending admin approval.",
        )
    if user["status"] != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account is {user['status']}.",
        )

    try:
        return await _issue_token_pair(user, refresh_repo)
    except Exception:
        logger.exception(
            "Failed to persist refresh session for user %s",
            user["username"],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create session. Please try again.",
        )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token_pair(
    request: RefreshRequest,
    user_repo: AuthRepository = Depends(get_auth_repository),
    refresh_repo=Depends(get_refresh_token_repository),
) -> TokenResponse:
    """Rotate a refresh token and return a new access/refresh pair.

    The presented refresh token is consumed (rotated). Reusing a rotated
    token is treated as suspected theft: the entire token family is
    revoked and the caller must log in again.
    """
    old_hash = hash_refresh_token(request.refresh_token)
    new_raw = create_refresh_token()
    new_hash = hash_refresh_token(new_raw)
    new_expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=REFRESH_TOKEN_TTL_SECONDS
    )

    result = await refresh_repo.rotate(old_hash, new_hash, new_expires_at)

    if result.status != RotationStatus.OK:
        # All non-OK outcomes present the same client-facing message so
        # an attacker cannot tell "expired" from "reuse detected".
        if result.status == RotationStatus.REUSE_DETECTED:
            logger.warning(
                "Refresh token reuse detected at endpoint; family revoked."
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token. Please log in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await user_repo.get_user_by_id(result.new_record.user_id)

    if user is None or user["status"] != "active":
        # The user was deleted or suspended between login and refresh.
        # Revoke the token we just issued so it cannot be used again.
        await refresh_repo.revoke_session(new_hash)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is no longer active.",
        )

    new_access = create_access_token(
        user_id=user["id"],
        role=user["role"],
    )

    return TokenResponse(
        access_token=new_access,
        refresh_token=new_raw,
        expires_in=ACCESS_TOKEN_TTL_SECONDS,
        user=_to_profile(user),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout_session(
    request: RefreshRequest,
    refresh_repo=Depends(get_refresh_token_repository),
) -> None:
    """Revoke the supplied refresh token.

    Always returns 204, even when the token is unknown or already
    revoked, so an attacker cannot probe which tokens exist. The
    matching access token (if still valid) is left alone; it will
    expire on its own within 15 minutes.
    """
    token_hash = hash_refresh_token(request.refresh_token)
    await refresh_repo.revoke_session(token_hash)


@router.get("/me", response_model=UserProfileResponse)
async def read_current_user(
    user: UserRecord = Depends(get_current_user_record),
) -> UserProfileResponse:
    """Return the currently authenticated user's profile."""
    return _to_profile(user)


# ---------------------------------------------------------------- #
# Admin-only endpoints                                             #
# ---------------------------------------------------------------- #


@router.get(
    "/admin/users",
    response_model=list[UserProfileResponse],
)
async def list_users(
    status_filter: str | None = None,
    repo: AuthRepository = Depends(get_auth_repository),
    _admin: UserRecord = Depends(require_admin),
) -> list[UserProfileResponse]:
    """List users, optionally filtered by status."""
    try:
        users = await repo.list_users(status=status_filter)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return [_to_profile(u) for u in users]


@router.post(
    "/admin/users/{user_id}/approve",
    response_model=UserProfileResponse,
)
async def approve_user(
    user_id: str,
    repo: AuthRepository = Depends(get_auth_repository),
    _admin: UserRecord = Depends(require_admin),
) -> UserProfileResponse:
    """Activate a pending account."""
    target = await repo.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    await repo.update_user_status(user_id, "active")
    updated = await repo.get_user_by_id(user_id)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User disappeared after approval.",
        )

    logger.info(
        "Admin %s approved user %s",
        _admin["username"], target["username"],
    )
    return _to_profile(updated)


@router.post(
    "/admin/users/{user_id}/suspend",
    response_model=UserProfileResponse,
)
async def suspend_user(
    user_id: str,
    repo: AuthRepository = Depends(get_auth_repository),
    admin: UserRecord = Depends(require_admin),
) -> UserProfileResponse:
    """Suspend an active account. Effect is immediate on next request."""
    if user_id == admin["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot suspend your own account.",
        )

    target = await repo.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    await repo.update_user_status(user_id, "suspended")
    updated = await repo.get_user_by_id(user_id)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User disappeared after suspension.",
        )

    logger.info(
        "Admin %s suspended user %s",
        admin["username"], target["username"],
    )
    return _to_profile(updated)


@router.delete(
    "/admin/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_user(
    user_id: str,
    repo: AuthRepository = Depends(get_auth_repository),
    admin: UserRecord = Depends(require_admin),
) -> None:
    """Permanently delete an account. Cannot delete yourself."""
    if user_id == admin["id"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account.",
        )

    target = await repo.get_user_by_id(user_id)
    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )

    await repo.delete_user(user_id)
    logger.info(
        "Admin %s deleted user %s",
        admin["username"], target["username"],
    )