import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth.dependencies import (
    get_auth_repository,
    get_current_user_record,
    require_admin,
)
from app.auth.repository import AuthRepository, UserRecord
from app.auth.security import (
    ACCESS_TOKEN_TTL_SECONDS,
    create_access_token,
    get_dummy_hash,
    hash_password,
    verify_password,
)
from app.models.auth_schemas import (
    TokenResponse,
    UserLoginRequest,
    UserProfileResponse,
    UserRegisterRequest,
)


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


def _to_profile(user: UserRecord) -> UserProfileResponse:
    """Convert an internal record into a public profile.

    We NEVER include `password_hash` in API responses. Keeping this
    conversion in one place prevents future code from accidentally
    serialising the full record.
    """
    return UserProfileResponse(
        id=user["id"],
        username=user["username"],
        role=user["role"],
        status=user["status"],
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
        # Returning 409 here leaks whether a username exists. That is
        # acceptable for registration (unlike login) because an attacker
        # can probe the same information just by trying to register.
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
        # Race condition: two concurrent registrations for the same
        # username. The DB unique constraint fires; surface as 409.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken.",
        ) from exc

    return _to_profile(new_user)


@router.post("/login", response_model=TokenResponse)
async def login_user(
    request: UserLoginRequest,
    repo: AuthRepository = Depends(get_auth_repository),
) -> TokenResponse:
    """Authenticate and issue a JWT for active accounts."""
    user = await repo.get_user_by_username(request.username)

    # Timing-attack mitigation: ALWAYS run verify_password, even when
    # the user does not exist. Otherwise an attacker could measure
    # response time to enumerate valid usernames.
    hash_to_check = user["password_hash"] if user else get_dummy_hash()
    password_ok = verify_password(request.password, hash_to_check)

    if not user or not password_ok:
        # Same generic message for both cases: never reveal which
        # part (username or password) was wrong.
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

    token = create_access_token(user_id=user["id"], role=user["role"])

    return TokenResponse(
        access_token=token,
        expires_in=ACCESS_TOKEN_TTL_SECONDS,
        user=_to_profile(user),
    )


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
    """List users, optionally filtered by status.

    Example: GET /api/v1/auth/admin/users?status_filter=pending
    """
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

    logger.info("Admin %s approved user %s", _admin["username"], target["username"])
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
    """Suspend an active account. Suspension takes effect immediately."""
    if user_id == admin["id"]:
        # Self-suspension would lock the admin out of their own system.
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

    logger.info("Admin %s suspended user %s", admin["username"], target["username"])
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
    logger.info("Admin %s deleted user %s", admin["username"], target["username"])