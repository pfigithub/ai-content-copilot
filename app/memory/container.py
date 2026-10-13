from dataclasses import dataclass

from fastapi import Request
from langgraph.store.base import BaseStore

from app.memory.commands import MemoryCommandService
from app.memory.long_term_memory import UserMemoryStore
from app.memory.proposal_service import MemoryProposalService
from app.memory.write_policy import MemoryWritePolicy


@dataclass
class MemoryServices:
    store: BaseStore
    user_memory_store: UserMemoryStore
    write_policy: MemoryWritePolicy
    command_service: MemoryCommandService
    proposal_service: MemoryProposalService


def build_memory_services(
    store: BaseStore,
) -> MemoryServices:
    user_memory_store = UserMemoryStore(
        store=store
    )

    write_policy = MemoryWritePolicy(
        memory_store=user_memory_store
    )

    command_service = MemoryCommandService(
        write_policy=write_policy
    )

    proposal_service = MemoryProposalService(
        command_service=command_service
    )

    return MemoryServices(
        store=store,
        user_memory_store=user_memory_store,
        write_policy=write_policy,
        command_service=command_service,
        proposal_service=proposal_service,
    )


def get_memory_proposal_service(
    request: Request,
) -> MemoryProposalService:
    return request.app.state.memory_services.proposal_service


def get_user_memory_store(
    request: Request,
) -> UserMemoryStore:
    return request.app.state.memory_services.user_memory_store