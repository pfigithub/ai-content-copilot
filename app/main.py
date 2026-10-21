
import os

# Enforce strict checkpoint deserialization before importing LangGraph.
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "true"

from contextlib import AsyncExitStack, asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.api.v1.endpoints.content import router as content_router
from app.api.v1.endpoints.memory import router as memory_router
from app.api.v1.endpoints.ingestion import (
    router as ingestion_router,
)
from app.api.v1.endpoints.rag import (
    router as rag_router,
)
from app.exception_handlers import (
    app_error_handler,
    validation_error_handler,
)
from app.exceptions import AppError
from app.memory.container import build_memory_services
from app.memory.postgres_config import get_postgres_uri
from app.services.document_ingestion_service import (
    DocumentIngestionService,
)
from app.rag.rag_graph import RAGGraph
from app.rag.research_query_planner import ResearchQueryPlanner
from app.rag.evidence_sufficiency import (
    EvidenceSufficiencyEvaluator,
)


def configure_app_resources(
    application: FastAPI,
    store: BaseStore,
    checkpointer,
) -> None:
    """Create and register application-wide service dependencies."""

    memory_services = build_memory_services(
        store=store
    )

    application.state.store = store
    application.state.checkpointer = checkpointer
    application.state.memory_services = memory_services

    # Create the RAG graph once so ingestion can share its vector store.
    rag_graph = RAGGraph(
        store=store,
        checkpointer=checkpointer,
        query_planner=ResearchQueryPlanner(),
        evidence_evaluator=EvidenceSufficiencyEvaluator(),
    )

    application.state.rag_graph = rag_graph

    # Ingested chunks must go into the same collection used by retrieval.
    application.state.document_ingestion_service = (
        DocumentIngestionService(
            store=rag_graph.retriever.store
        )
    )

def create_app(
    use_in_memory: bool | None = None,
) -> FastAPI:
    if use_in_memory is None:
        use_in_memory = (
            os.getenv("APP_ENV", "development").strip().lower()
            == "test"
        )

    @asynccontextmanager
    async def lifespan(
        application: FastAPI,
    ) -> AsyncIterator[None]:
        if use_in_memory:
            store = InMemoryStore()
            checkpointer = InMemorySaver()

            configure_app_resources(
                application=application,
                store=store,
                checkpointer=checkpointer,
            )

            yield
            return

        postgres_uri = get_postgres_uri()

        async with AsyncExitStack() as stack:
            store = await stack.enter_async_context(
                AsyncPostgresStore.from_conn_string(
                    postgres_uri
                )
            )

            checkpointer = await stack.enter_async_context(
                AsyncPostgresSaver.from_conn_string(
                    postgres_uri
                )
            )

            await store.setup()
            await checkpointer.setup()

            configure_app_resources(
                application=application,
                store=store,
                checkpointer=checkpointer,
            )

            # Keep both resources open for the application lifetime.
            yield

        # AsyncExitStack closes the resources during shutdown.

    application = FastAPI(
        title="AI Content Copilot",
        version="0.1.0.dev0",
        lifespan=lifespan,
    )

    application.add_exception_handler(
        AppError,
        app_error_handler,
    )

    application.add_exception_handler(
        RequestValidationError,
        validation_error_handler,
    )

    application.include_router(content_router)
    application.include_router(memory_router)
    application.include_router(ingestion_router)
    application.include_router(rag_router)

    @application.get("/")
    async def root():
        return {
            "status": "ok",
            "service": "AI Content Copilot",
            "version": "0.1.0.dev0",
        }

    return application


# Normal development execution uses PostgreSQL.
# The test environment can select in-memory resources.
app = create_app()
