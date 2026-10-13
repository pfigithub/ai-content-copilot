import asyncio
import selectors
from typing import TypedDict
from uuid import uuid4

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.memory.postgres_config import get_postgres_uri


class SmokeTestState(TypedDict):
    value: str


async def verify_store(
    store: AsyncPostgresStore,
) -> None:
    namespace = (
        "setup_check",
        "local",
    )
    key = str(uuid4())
    expected = {
        "status": "ok",
        "component": "AsyncPostgresStore",
    }

    await store.aput(
        namespace,
        key,
        expected,
        index=False,
    )

    try:
        item = await store.aget(
            namespace,
            key,
        )

        if item is None or item.value != expected:
            raise RuntimeError(
                "PostgreSQL Store read/write verification failed."
            )
    finally:
        await store.adelete(
            namespace,
            key,
        )

    print("AsyncPostgresStore: setup and read/write OK")


async def verify_checkpointer(
    checkpointer: AsyncPostgresSaver,
) -> None:
    thread_id = f"postgres-smoke-{uuid4()}"

    builder = StateGraph(SmokeTestState)

    builder.add_node(
        "verify",
        lambda state: {
            "value": state["value"],
        },
    )

    builder.add_edge(
        START,
        "verify",
    )

    builder.add_edge(
        "verify",
        END,
    )

    graph = builder.compile(
        checkpointer=checkpointer,
    )

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    try:
        result = await graph.ainvoke(
            {
                "value": "persistent",
            },
            config,
        )

        snapshot = await graph.aget_state(
            config,
        )

        if result.get("value") != "persistent":
            raise RuntimeError(
                "Graph execution verification failed."
            )

        if snapshot.values.get("value") != "persistent":
            raise RuntimeError(
                "PostgreSQL checkpoint read verification failed."
            )

    finally:
        await checkpointer.adelete_thread(
            thread_id,
        )

    print("AsyncPostgresSaver: setup and checkpoint read/write OK")


async def main() -> None:
    postgres_uri = get_postgres_uri()

    async with AsyncPostgresStore.from_conn_string(
        postgres_uri,
    ) as store:
        await store.setup()
        await verify_store(store)

    async with AsyncPostgresSaver.from_conn_string(
        postgres_uri,
    ) as checkpointer:
        await checkpointer.setup()
        await verify_checkpointer(checkpointer)

    print("PostgreSQL initialization and smoke tests completed.")


if __name__ == "__main__":
    asyncio.run(
        main(),
        loop_factory=lambda: asyncio.SelectorEventLoop(
            selectors.SelectSelector()
        ),
    )