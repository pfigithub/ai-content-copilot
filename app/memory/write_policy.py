from dataclasses import dataclass
from typing import Literal

from app.memory.long_term_memory import UserMemoryStore


MemoryType = Literal[
    "preference",
    "self_reported_fact",
]

MAX_MEMORY_CONTENT_LENGTH = 500


@dataclass(frozen=True)
class ExplicitMemoryRequest:
    user_id: str
    memory_id: str
    content: str
    memory_type: MemoryType = "preference"
    confirmed_by_user: bool = False


class MemoryWritePolicy:
    def __init__(
        self,
        memory_store: UserMemoryStore,
    ):
        self.memory_store = memory_store

    async def write(
        self,
        request: ExplicitMemoryRequest,
    ) -> None:
        if not request.confirmed_by_user:
            raise PermissionError(
                "Explicit user confirmation is required "
                "before writing long-term memory."
            )

        content = request.content.strip()

        if not content:
            raise ValueError(
                "Memory content cannot be empty."
            )

        if (
            len(content)
            > MAX_MEMORY_CONTENT_LENGTH
        ):
            raise ValueError(
                "Memory content exceeds the maximum "
                "allowed length."
            )

        await self.memory_store.save_explicit(
            user_id=request.user_id,
            memory_id=request.memory_id,
            content=content,
            memory_type=request.memory_type,
        )