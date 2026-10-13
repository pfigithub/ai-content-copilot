import asyncio

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client():
    test_app = create_app(
        use_in_memory=True
    )

    with TestClient(test_app) as test_client:
        yield test_client


def get_test_memory(
    client: TestClient,
    user_id: str,
    memory_id: str,
):
    return asyncio.run(
        client.app.state.memory_services.user_memory_store.get(
            user_id=user_id,
            memory_id=memory_id,
        )
    )


def test_proposal_does_not_write_until_confirmed(client):
    user_id = "api-test-proposal-user"
    memory_id = "api-test-language"

    proposal_response = client.post(
        "/api/v1/memory/proposals",
        json={
            "user_id": user_id,
            "memory_id": memory_id,
            "content": "کاربر پاسخ‌های فارسی می‌خواهد.",
            "memory_type": "preference",
        },
    )

    assert proposal_response.status_code == 201

    proposal = proposal_response.json()

    assert proposal["confirmation_required"] is True

    assert get_test_memory(
        client,
        user_id,
        memory_id,
    ) is None


def test_confirmed_proposal_is_stored(client):
    user_id = "api-test-confirm-user"
    memory_id = "api-test-confirm-language"

    proposal_response = client.post(
        "/api/v1/memory/proposals",
        json={
            "user_id": user_id,
            "memory_id": memory_id,
            "content": "کاربر پاسخ‌های فارسی می‌خواهد.",
            "memory_type": "preference",
        },
    )

    proposal_id = proposal_response.json()["proposal_id"]

    confirm_response = client.post(
        f"/api/v1/memory/proposals/{proposal_id}/confirm",
        json={
            "user_id": user_id,
        },
    )

    assert confirm_response.status_code == 200
    assert confirm_response.json()["status"] == "stored"

    memory = get_test_memory(
        client,
        user_id,
        memory_id,
    )

    assert memory is not None
    assert memory["content"] == (
        "کاربر پاسخ‌های فارسی می‌خواهد."
    )


def test_another_user_cannot_confirm_proposal(client):
    owner_id = "api-test-owner"
    other_id = "api-test-other"
    memory_id = "api-test-private-language"

    proposal_response = client.post(
        "/api/v1/memory/proposals",
        json={
            "user_id": owner_id,
            "memory_id": memory_id,
            "content": "کاربر پاسخ‌های فارسی می‌خواهد.",
            "memory_type": "preference",
        },
    )

    proposal_id = proposal_response.json()["proposal_id"]

    response = client.post(
        f"/api/v1/memory/proposals/{proposal_id}/confirm",
        json={
            "user_id": other_id,
        },
    )

    assert response.status_code == 404

    assert get_test_memory(
        client,
        owner_id,
        memory_id,
    ) is None


def test_rejected_proposal_is_not_stored(client):
    user_id = "api-test-reject-user"
    memory_id = "api-test-rejected-memory"

    proposal_response = client.post(
        "/api/v1/memory/proposals",
        json={
            "user_id": user_id,
            "memory_id": memory_id,
            "content": "کاربر پاسخ‌های فارسی می‌خواهد.",
            "memory_type": "preference",
        },
    )

    proposal_id = proposal_response.json()["proposal_id"]

    reject_response = client.post(
        f"/api/v1/memory/proposals/{proposal_id}/reject",
        json={
            "user_id": user_id,
        },
    )

    assert reject_response.status_code == 200
    assert reject_response.json()["status"] == "rejected"

    assert get_test_memory(
        client,
        user_id,
        memory_id,
    ) is None


def test_confirmed_proposal_cannot_be_reused(client):
    user_id = "api-test-replay-user"
    memory_id = "api-test-replay-memory"

    proposal_response = client.post(
        "/api/v1/memory/proposals",
        json={
            "user_id": user_id,
            "memory_id": memory_id,
            "content": "کاربر پاسخ‌های فارسی می‌خواهد.",
            "memory_type": "preference",
        },
    )

    proposal_id = proposal_response.json()["proposal_id"]

    first = client.post(
        f"/api/v1/memory/proposals/{proposal_id}/confirm",
        json={
            "user_id": user_id,
        },
    )

    second = client.post(
        f"/api/v1/memory/proposals/{proposal_id}/confirm",
        json={
            "user_id": user_id,
        },
    )

    assert first.status_code == 200
    assert second.status_code == 404


def test_preferences_can_be_listed_and_deleted(client):
    user_id = "api-test-delete-user"
    memory_id = "api-test-delete-memory"

    proposal_response = client.post(
        "/api/v1/memory/proposals",
        json={
            "user_id": user_id,
            "memory_id": memory_id,
            "content": "کاربر پاسخ‌های فارسی می‌خواهد.",
            "memory_type": "preference",
        },
    )

    proposal_id = proposal_response.json()["proposal_id"]

    client.post(
        f"/api/v1/memory/proposals/{proposal_id}/confirm",
        json={
            "user_id": user_id,
        },
    )

    preferences_response = client.get(
        f"/api/v1/memory/users/{user_id}/preferences"
    )

    assert preferences_response.status_code == 200
    assert len(
        preferences_response.json()["preferences"]
    ) == 1

    delete_response = client.delete(
        f"/api/v1/memory/users/{user_id}/memories/{memory_id}"
    )

    assert delete_response.status_code == 200
    assert delete_response.json()["status"] == "deleted"

    assert get_test_memory(
        client,
        user_id,
        memory_id,
    ) is None