from langgraph.store.memory import InMemoryStore

from app.memory.commands import MemoryCommandService
from app.memory.long_term_memory import UserMemoryStore
from app.memory.proposal_service import MemoryProposalService
from app.memory.write_policy import MemoryWritePolicy


# One shared store for the application process.
shared_memory_store = InMemoryStore()

user_memory_store = UserMemoryStore(
    store=shared_memory_store
)

memory_write_policy = MemoryWritePolicy(
    memory_store=user_memory_store
)

memory_command_service = MemoryCommandService(
    write_policy=memory_write_policy
)

memory_proposal_service = MemoryProposalService(
    command_service=memory_command_service
)


def get_memory_proposal_service():
    return memory_proposal_service


def get_user_memory_store():
    return user_memory_store