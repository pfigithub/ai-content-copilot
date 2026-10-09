from langgraph.store.memory import InMemoryStore

from app.memory.commands import MemoryCommandService
from app.memory.long_term_memory import UserMemoryStore
from app.memory.write_policy import MemoryWritePolicy


def create_service():
    store = UserMemoryStore(
        store=InMemoryStore()
    )

    policy = MemoryWritePolicy(
        memory_store=store
    )

    service = MemoryCommandService(
        write_policy=policy
    )

    return service, store


def test_propose_does_not_write_memory():
    service, store = create_service()

    proposal = service.propose(
        user_id="user-001",
        memory_id="language",
        content="کاربر پاسخ‌های فارسی می‌خواهد.",
    )

    assert proposal.confirmation_required is True

    assert (
        store.get(
            user_id="user-001",
            memory_id="language",
        )
        is None
    )


def test_confirm_writes_memory():
    service, store = create_service()

    proposal = service.propose(
        user_id="user-001",
        memory_id="language",
        content="کاربر پاسخ‌های فارسی می‌خواهد.",
    )

    service.confirm(proposal)

    memory = store.get(
        user_id="user-001",
        memory_id="language",
    )

    assert memory is not None

    assert memory["content"] == (
        "کاربر پاسخ‌های فارسی می‌خواهد."
    )

    assert memory["source"] == (
        "user_explicit"
    )


def test_proposal_preserves_memory_type():
    service, store = create_service()

    proposal = service.propose(
        user_id="user-001",
        memory_id="profession",
        content="کاربر برنامه‌نویس است.",
        memory_type="self_reported_fact",
    )

    assert (
        proposal.memory_type
        == "self_reported_fact"
    )

    assert (
        store.get(
            user_id="user-001",
            memory_id="profession",
        )
        is None
    )