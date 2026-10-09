from langgraph.store.memory import InMemoryStore

from app.memory.long_term_memory import UserMemoryStore
from app.memory.write_policy import (
    ExplicitMemoryRequest,
    MemoryWritePolicy,
)


def test_confirmed_memory_request_is_saved():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    policy = MemoryWritePolicy(
        memory_store=store
    )

    request = ExplicitMemoryRequest(
        user_id="user-001",
        memory_id="language",
        content="کاربر پاسخ‌های فارسی می‌خواهد.",
        memory_type="preference",
        confirmed_by_user=True,
    )

    policy.write(request)

    memory = store.get(
        user_id="user-001",
        memory_id="language",
    )

    assert memory is not None

    assert memory["content"] == (
        "کاربر پاسخ‌های فارسی می‌خواهد."
    )


def test_unconfirmed_memory_request_is_rejected():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    policy = MemoryWritePolicy(
        memory_store=store
    )

    request = ExplicitMemoryRequest(
        user_id="user-001",
        memory_id="language",
        content="کاربر پاسخ‌های فارسی می‌خواهد.",
        confirmed_by_user=False,
    )

    try:
        policy.write(request)
    except PermissionError as exc:
        assert str(exc) == (
            "Explicit user confirmation is required "
            "before writing long-term memory."
        )
    else:
        raise AssertionError(
            "Expected PermissionError"
        )

    assert (
        store.get(
            user_id="user-001",
            memory_id="language",
        )
        is None
    )


def test_memory_content_is_trimmed_before_saving():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    policy = MemoryWritePolicy(
        memory_store=store
    )

    request = ExplicitMemoryRequest(
        user_id="user-001",
        memory_id="language",
        content="   فارسی   ",
        confirmed_by_user=True,
    )

    policy.write(request)

    memory = store.get(
        user_id="user-001",
        memory_id="language",
    )

    assert memory["content"] == "فارسی"


def test_memory_content_length_is_limited():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    policy = MemoryWritePolicy(
        memory_store=store
    )

    request = ExplicitMemoryRequest(
        user_id="user-001",
        memory_id="large-memory",
        content="الف" * 501,
        confirmed_by_user=True,
    )

    try:
        policy.write(request)
    except ValueError as exc:
        assert str(exc) == (
            "Memory content exceeds the maximum "
            "allowed length."
        )
    else:
        raise AssertionError(
            "Expected ValueError"
        )