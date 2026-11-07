
import asyncio
import hashlib

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


TOKEN_A = "memory-api-test-token-user-a"
TOKEN_B = "memory-api-test-token-user-b"
USER_A = "memory_api_user_a"
USER_B = "memory_api_user_b"


def auth_headers(token: str = TOKEN_A) -> dict[str, str]:
    """Build the Bearer header used by an API test client."""
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    """Create an isolated app with two authenticated test identities."""
    monkeypatch.setattr(
        settings,
        "api_key_hashes",
        {
            hashlib.sha256(TOKEN_A.encode("utf-8")).hexdigest(): USER_A,
            hashlib.sha256(TOKEN_B.encode("utf-8")).hexdigest(): USER_B,
        },
    )

    test_app = create_app(use_in_memory=True)
    with TestClient(test_app) as test_client:
        yield test_client


def get_test_memory(client: TestClient, user_id: str, memory_id: str):
    """Read test state directly to verify persistence and tenant isolation."""
    return asyncio.run(
        client.app.state.memory_services.user_memory_store.get(
            user_id=user_id,
            memory_id=memory_id,
        )
    )


def create_proposal(
    client: TestClient,
    memory_id: str,
    token: str = TOKEN_A,
):
    """Create a proposal through the public authenticated API contract."""
    return client.post(
        "/api/v1/memory/proposals",
        json={
            "memory_id": memory_id,
            "content": "کاربر پاسخ‌های فارسی را ترجیح می‌دهد.",
            "memory_type": "preference",
        },
        headers=auth_headers(token),
    )


def confirm_proposal(client: TestClient, proposal_id: str, token: str = TOKEN_A):
    """Confirm a proposal using only the authenticated identity."""
    return client.post(
        f"/api/v1/memory/proposals/{proposal_id}/confirm",
        headers=auth_headers(token),
    )


def test_proposal_does_not_write_until_confirmed(client: TestClient):
    """A proposal remains pending until its owner explicitly confirms it."""
    memory_id = "api-test-language"
    response = create_proposal(client, memory_id)

    assert response.status_code == 201
    proposal = response.json()
    assert proposal["user_id"] == USER_A
    assert proposal["confirmation_required"] is True
    assert get_test_memory(client, USER_A, memory_id) is None


def test_proposal_rejects_client_supplied_user_id(client: TestClient):
    """The client must not be able to choose the owner in the request body."""
    response = client.post(
        "/api/v1/memory/proposals",
        json={
            "user_id": USER_B,
            "memory_id": "spoofed-owner-memory",
            "content": "This must not be accepted.",
            "memory_type": "preference",
        },
        headers=auth_headers(TOKEN_A),
    )

    assert response.status_code == 422


def test_confirmed_proposal_is_stored_for_authenticated_user(client: TestClient):
    """A valid confirmation stores the proposal in the owner's namespace."""
    memory_id = "api-test-confirm-language"
    proposal_response = create_proposal(client, memory_id)
    assert proposal_response.status_code == 201

    response = confirm_proposal(
        client,
        proposal_response.json()["proposal_id"],
    )

    assert response.status_code == 200
    assert response.json()["status"] == "stored"
    memory = get_test_memory(client, USER_A, memory_id)
    assert memory is not None
    assert memory["content"] == "کاربر پاسخ‌های فارسی را ترجیح می‌دهد."


def test_another_user_cannot_confirm_proposal(client: TestClient):
    """A different authenticated user cannot confirm another user's proposal."""
    memory_id = "api-test-private-language"
    proposal_response = create_proposal(client, memory_id)
    proposal_id = proposal_response.json()["proposal_id"]

    response = confirm_proposal(client, proposal_id, TOKEN_B)

    assert response.status_code == 404
    assert get_test_memory(client, USER_A, memory_id) is None
    assert get_test_memory(client, USER_B, memory_id) is None


def test_another_user_cannot_reject_proposal(client: TestClient):
    """A rejected cross-user attempt leaves the owner's proposal available."""
    memory_id = "api-test-private-rejection"
    proposal_response = create_proposal(client, memory_id)
    proposal_id = proposal_response.json()["proposal_id"]

    response = client.post(
        f"/api/v1/memory/proposals/{proposal_id}/reject",
        headers=auth_headers(TOKEN_B),
    )
    owner_response = confirm_proposal(client, proposal_id, TOKEN_A)

    assert response.status_code == 404
    assert owner_response.status_code == 200
    assert get_test_memory(client, USER_A, memory_id) is not None


def test_rejected_proposal_is_not_stored(client: TestClient):
    """Rejecting a proposal removes it without writing a memory record."""
    memory_id = "api-test-rejected-memory"
    proposal_response = create_proposal(client, memory_id)
    proposal_id = proposal_response.json()["proposal_id"]

    response = client.post(
        f"/api/v1/memory/proposals/{proposal_id}/reject",
        headers=auth_headers(TOKEN_A),
    )

    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert get_test_memory(client, USER_A, memory_id) is None


def test_confirmed_proposal_cannot_be_reused(client: TestClient):
    """A successfully confirmed proposal is single-use."""
    memory_id = "api-test-replay-memory"
    proposal_response = create_proposal(client, memory_id)
    proposal_id = proposal_response.json()["proposal_id"]

    first = confirm_proposal(client, proposal_id)
    second = confirm_proposal(client, proposal_id)

    assert first.status_code == 200
    assert second.status_code == 404


def test_preferences_are_scoped_to_authenticated_user(client: TestClient):
    """Listing preferences never reads from another user's namespace."""
    memory_id = "api-test-user-isolation"
    proposal_response = create_proposal(client, memory_id)
    assert confirm_proposal(
        client,
        proposal_response.json()["proposal_id"],
    ).status_code == 200

    owner_response = client.get(
        "/api/v1/memory/preferences",
        headers=auth_headers(TOKEN_A),
    )
    other_response = client.get(
        "/api/v1/memory/preferences",
        headers=auth_headers(TOKEN_B),
    )

    assert owner_response.status_code == 200
    assert owner_response.json()["user_id"] == USER_A
    assert len(owner_response.json()["preferences"]) == 1
    assert other_response.status_code == 200
    assert other_response.json()["user_id"] == USER_B
    assert other_response.json()["preferences"] == []


def test_user_cannot_delete_another_users_memory(client: TestClient):
    """Deleting a matching ID in user B's namespace leaves user A's record."""
    memory_id = "api-test-cross-user-delete"
    proposal_response = create_proposal(client, memory_id)
    assert confirm_proposal(
        client,
        proposal_response.json()["proposal_id"],
    ).status_code == 200

    response = client.delete(
        f"/api/v1/memory/memories/{memory_id}",
        headers=auth_headers(TOKEN_B),
    )

    assert response.status_code == 200
    assert get_test_memory(client, USER_A, memory_id) is not None
    assert get_test_memory(client, USER_B, memory_id) is None


def test_preferences_can_be_listed_and_deleted(client: TestClient):
    """The authenticated owner can list and delete their own memory."""
    memory_id = "api-test-delete-memory"
    proposal_response = create_proposal(client, memory_id)
    assert confirm_proposal(
        client,
        proposal_response.json()["proposal_id"],
    ).status_code == 200

    preferences_response = client.get(
        "/api/v1/memory/preferences",
        headers=auth_headers(TOKEN_A),
    )
    delete_response = client.delete(
        f"/api/v1/memory/memories/{memory_id}",
        headers=auth_headers(TOKEN_A),
    )

    assert preferences_response.status_code == 200
    assert len(preferences_response.json()["preferences"]) == 1
    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"
    assert get_test_memory(client, USER_A, memory_id) is None


def test_memory_endpoints_require_authentication(client: TestClient):
    """Memory endpoints reject requests that do not carry a Bearer token."""
    response = client.get("/api/v1/memory/preferences")
    assert response.status_code == 401
