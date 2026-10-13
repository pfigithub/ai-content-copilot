import asyncio

from langgraph.store.memory import InMemoryStore

from app.memory.long_term_memory import UserMemoryStore


def test_user_memory_can_be_saved_and_retrieved():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    asyncio.run(
        store.save_explicit(
            user_id="user-001",
            memory_id="response-language",
            content="کاربر پاسخ‌های فارسی می‌خواهد.",
        )
    )

    memory = asyncio.run(
        store.get(
            user_id="user-001",
            memory_id="response-language",
        )
    )

    assert memory is not None
    assert memory["content"] == (
        "کاربر پاسخ‌های فارسی می‌خواهد."
    )
    assert memory["source"] == "user_explicit"
    assert memory["memory_type"] == "preference"


def test_user_memories_are_isolated():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    asyncio.run(
        store.save_explicit(
            user_id="user-001",
            memory_id="language",
            content="فارسی",
        )
    )

    asyncio.run(
        store.save_explicit(
            user_id="user-002",
            memory_id="language",
            content="English",
        )
    )

    memory_1 = asyncio.run(
        store.get(
            user_id="user-001",
            memory_id="language",
        )
    )

    memory_2 = asyncio.run(
        store.get(
            user_id="user-002",
            memory_id="language",
        )
    )

    assert memory_1 is not None
    assert memory_2 is not None
    assert memory_1["content"] == "فارسی"
    assert memory_2["content"] == "English"


def test_unknown_memory_returns_none():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    memory = asyncio.run(
        store.get(
            user_id="user-001",
            memory_id="unknown",
        )
    )

    assert memory is None


def test_empty_memory_content_is_rejected():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    try:
        asyncio.run(
            store.save_explicit(
                user_id="user-001",
                memory_id="test",
                content="",
            )
        )
    except ValueError as exc:
        assert str(exc) == "Memory content cannot be empty."
    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_preferences_exclude_self_reported_facts():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    asyncio.run(
        store.save_explicit(
            user_id="user-001",
            memory_id="language",
            content="کاربر پاسخ‌های فارسی می‌خواهد.",
            memory_type="preference",
        )
    )

    asyncio.run(
        store.save_explicit(
            user_id="user-001",
            memory_id="profession",
            content="کاربر برنامه‌نویس است.",
            memory_type="self_reported_fact",
        )
    )

    preferences = asyncio.run(
        store.get_preferences(
            user_id="user-001"
        )
    )

    assert len(preferences) == 1
    assert preferences[0]["content"] == (
        "کاربر پاسخ‌های فارسی می‌خواهد."
    )
    assert preferences[0]["memory_type"] == "preference"


def test_existing_memory_is_replaced():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    asyncio.run(
        store.save_explicit(
            user_id="user-001",
            memory_id="response-style",
            content="پاسخ کوتاه",
        )
    )

    asyncio.run(
        store.save_explicit(
            user_id="user-001",
            memory_id="response-style",
            content="پاسخ کامل",
        )
    )

    memory = asyncio.run(
        store.get(
            user_id="user-001",
            memory_id="response-style",
        )
    )

    assert memory is not None
    assert memory["content"] == "پاسخ کامل"


def test_memory_can_be_deleted():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    asyncio.run(
        store.save_explicit(
            user_id="user-001",
            memory_id="response-style",
            content="پاسخ کوتاه",
        )
    )

    asyncio.run(
        store.delete(
            user_id="user-001",
            memory_id="response-style",
        )
    )

    memory = asyncio.run(
        store.get(
            user_id="user-001",
            memory_id="response-style",
        )
    )

    assert memory is None