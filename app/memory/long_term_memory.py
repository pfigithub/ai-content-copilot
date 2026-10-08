from typing import Any, Literal

from langgraph.store.memory import InMemoryStore


MemoryType = Literal[
    "preference",
    "self_reported_fact",
]

MAX_USER_PREFERENCES = 10


class UserMemoryStore:
    def __init__(
        self,
        store: InMemoryStore | None = None,
    ):
        self.store = (
            store
            if store is not None
            else InMemoryStore()
        )

    @staticmethod
    def _validate_user_id(
        user_id: str,
    ) -> None:
        if not user_id or not user_id.strip():
            raise ValueError(
                "user_id cannot be empty."
            )

    @staticmethod
    def _validate_memory_id(
        memory_id: str,
    ) -> None:
        if not memory_id or not memory_id.strip():
            raise ValueError(
                "memory_id cannot be empty."
            )

    @staticmethod
    def _validate_content(
        content: str,
    ) -> None:
        if not content or not content.strip():
            raise ValueError(
                "Memory content cannot be empty."
            )

    @staticmethod
    def _namespace(
        user_id: str,
    ) -> tuple[str, str]:
        return (
            "memories",
            user_id,
        )

    def save_explicit(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        memory_type: MemoryType = "preference",
    ) -> None:
        self._validate_user_id(user_id)
        self._validate_memory_id(memory_id)
        self._validate_content(content)

        self.store.put(
            self._namespace(user_id),
            memory_id,
            {
                "content": content.strip(),
                "source": "user_explicit",
                "memory_type": memory_type,
            },
            index=False,
        )

    def get(
        self,
        user_id: str,
        memory_id: str,
    ) -> dict[str, Any] | None:
        self._validate_user_id(user_id)
        self._validate_memory_id(memory_id)

        item = self.store.get(
            self._namespace(user_id),
            memory_id,
        )

        if item is None:
            return None

        return item.value

    def delete(
        self,
        user_id: str,
        memory_id: str,
    ) -> None:
        self._validate_user_id(user_id)
        self._validate_memory_id(memory_id)

        self.store.delete(
            self._namespace(user_id),
            memory_id,
        )

    def get_preferences(
        self,
        user_id: str,
        limit: int = MAX_USER_PREFERENCES,
    ) -> list[dict[str, Any]]:
        self._validate_user_id(user_id)

        if limit <= 0:
            raise ValueError(
                "limit must be greater than zero."
            )

        items = self.store.search(
            self._namespace(user_id),
            filter={
                "memory_type": "preference",
            },
            limit=limit,
        )

        return [
            item.value
            for item in items
        ]