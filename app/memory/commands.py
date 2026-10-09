from dataclasses import dataclass
from typing import Literal

from app.memory.write_policy import (
    ExplicitMemoryRequest,
    MemoryWritePolicy,
)


MemoryType = Literal[
    "preference",
    "self_reported_fact",
]


@dataclass(frozen=True)
class MemoryProposal:
    user_id: str
    memory_id: str
    content: str
    memory_type: MemoryType
    confirmation_required: bool = True


class MemoryCommandService:
    def __init__(
        self,
        write_policy: MemoryWritePolicy,
    ):
        self.write_policy = write_policy

    def propose(
        self,
        user_id: str,
        memory_id: str,
        content: str,
        memory_type: MemoryType = "preference",
    ) -> MemoryProposal:
        if not user_id or not user_id.strip():
            raise ValueError(
                "user_id cannot be empty."
            )

        if not memory_id or not memory_id.strip():
            raise ValueError(
                "memory_id cannot be empty."
            )

        if not content or not content.strip():
            raise ValueError(
                "Memory content cannot be empty."
            )

        return MemoryProposal(
            user_id=user_id,
            memory_id=memory_id,
            content=content.strip(),
            memory_type=memory_type,
        )

    def confirm(
        self,
        proposal: MemoryProposal,
    ) -> None:
        request = ExplicitMemoryRequest(
            user_id=proposal.user_id,
            memory_id=proposal.memory_id,
            content=proposal.content,
            memory_type=proposal.memory_type,
            confirmed_by_user=True,
        )

        self.write_policy.write(request)