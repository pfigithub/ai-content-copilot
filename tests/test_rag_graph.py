import asyncio

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langgraph.store.memory import InMemoryStore

from app.memory.long_term_memory import UserMemoryStore
from app.rag.rag_chain import RAGAnswer, RAGChain
from app.rag.rag_graph import RAGGraph


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = 0
        self.queries = []

    async def retrieve(
        self,
        query,
        k=4,
        score_threshold=None,
    ):
        self.calls += 1
        self.queries.append(query)

        return self.results


class FakeModel:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, _schema):
        async def invoke(_input):
            self.calls += 1

            return RAGAnswer(
                answer=(
                    "پاسخ آزمایشی بر اساس "
                    "منابع بازیابی‌شده"
                )
            )

        return RunnableLambda(invoke)


class FakeContextualizer:
    def __init__(self):
        self.calls = 0
        self.questions = []
        self.histories = []

    async def contextualize(
        self,
        question,
        history=None,
    ):
        self.calls += 1
        self.questions.append(question)
        self.histories.append(history or [])

        return (
            "مزایای آبیاری قطره‌ای "
            "در کاهش مصرف آب کشاورزی چیست؟"
        )


def test_rag_graph_retrieves_once_and_generates_from_same_results():
    document = Document(
        page_content=(
            "آبیاری قطره‌ای می‌تواند "
            "مصرف آب را کاهش دهد."
        ),
        metadata={
            "source": "agriculture.txt",
            "source_type": "evaluation",
            "chunk_index": 2,
            "chunk_count": 5,
        },
    )

    retriever = FakeRetriever(
        results=[
            (document, 0.72),
        ]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    response = asyncio.run(
        graph.answer(
            query="چطور مصرف آب کشاورزی را کاهش دهیم؟"
        )
    )

    assert retriever.calls == 1
    assert model.calls == 1

    assert response.answer == (
        "پاسخ آزمایشی بر اساس "
        "منابع بازیابی‌شده"
    )

    assert len(response.sources) == 1

    source = response.sources[0]

    assert source.source == "agriculture.txt"
    assert source.source_type == "evaluation"
    assert source.chunk_index == 2
    assert source.chunk_count == 5
    assert source.score == 0.72


def test_rag_graph_abstains_without_results():
    retriever = FakeRetriever(
        results=[]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    response = asyncio.run(
        graph.answer(
            query="سؤال خارج از منابع"
        )
    )

    assert retriever.calls == 1
    assert model.calls == 0

    assert response.sources == []

    assert response.answer == (
        "اطلاعات مرتبط و کافی در منابع "
        "بازیابی‌شده پیدا نشد."
    )


def test_rag_graph_streams_retrieve_and_generate_updates():
    document = Document(
        page_content=(
            "آبیاری قطره‌ای می‌تواند "
            "مصرف آب را کاهش دهد."
        ),
        metadata={
            "source": "agriculture.txt",
            "source_type": "evaluation",
            "chunk_index": 2,
            "chunk_count": 5,
        },
    )

    retriever = FakeRetriever(
        results=[
            (document, 0.72),
        ]
    )

    model = FakeModel()

    contextualizer = FakeContextualizer()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
        contextualizer=contextualizer,
    )

    updates = []

    async def collect_updates():
        async for update in graph.stream_steps(
            query="چطور مصرف آب کشاورزی را کاهش دهیم؟"
        ):
            updates.append(update)

    asyncio.run(collect_updates())

    update_nodes = [
        next(iter(update.keys()))
        for update in updates
        if update
    ]

    assert update_nodes[:4] == [
        "load_memory",
        "contextualize",
        "retrieve",
        "generate",
    ]

    assert (
        "retrieval_query"
        in updates[1]["contextualize"]
    )

    assert (
        "results"
        in updates[2]["retrieve"]
    )

    assert (
        "response"
        in updates[3]["generate"]
    )

    assert retriever.calls == 1
    assert model.calls == 1

def test_rag_graph_persists_state_by_thread():
    retriever = FakeRetriever(
        results=[]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    thread_id = "thread-persistence-test"

    response = asyncio.run(
        graph.answer(
            query="سؤال آزمایشی برای persistence",
            thread_id=thread_id,
        )
    )

    assert response.sources == []

    snapshot = graph.get_state(thread_id)

    assert snapshot.values["query"] == (
        "سؤال آزمایشی برای persistence"
    )

    assert snapshot.values["results"] == []

    assert snapshot.values["response"]["answer"] == (
        "اطلاعات مرتبط و کافی در منابع "
        "بازیابی‌شده پیدا نشد."
    )


def test_rag_graph_keeps_threads_separate():
    retriever = FakeRetriever(
        results=[]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
    )

    asyncio.run(
        graph.answer(
            query="سؤال مربوط به thread اول",
            thread_id="thread-1",
        )
    )

    asyncio.run(
        graph.answer(
            query="سؤال مربوط به thread دوم",
            thread_id="thread-2",
        )
    )

    state_1 = graph.get_state(
        "thread-1"
    )

    state_2 = graph.get_state(
        "thread-2"
    )

    assert state_1.values["query"] == (
        "سؤال مربوط به thread اول"
    )

    assert state_2.values["query"] == (
        "سؤال مربوط به thread دوم"
    )


def test_rag_graph_keeps_conversation_history():
    retriever = FakeRetriever(
        results=[]
    )

    model = FakeModel()
    
    contextualizer = FakeContextualizer()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
        contextualizer=contextualizer,
    )

    thread_id = "conversation-memory-test"

    asyncio.run(
        graph.answer(
            query="پیام اول",
            thread_id=thread_id,
        )
    )

    asyncio.run(
        graph.answer(
            query="پیام دوم",
            thread_id=thread_id,
        )
    )

    snapshot = graph.get_state(
        thread_id
    )

    messages = snapshot.values["messages"]

    assert len(messages) == 4

    assert messages[0].type == "human"
    assert messages[0].content == "پیام اول"

    assert messages[1].type == "ai"

    assert messages[2].type == "human"
    assert messages[2].content == "پیام دوم"

    assert messages[3].type == "ai"
    assert contextualizer.calls == 1


def test_rag_graph_uses_contextualized_query_for_retrieval():
    document = Document(
        page_content=(
            "آبیاری قطره‌ای یکی از روش‌های "
            "کاهش مصرف آب کشاورزی است."
        ),
        metadata={
            "source": "agriculture.txt",
            "source_type": "evaluation",
            "chunk_index": 1,
            "chunk_count": 3,
        },
    )

    retriever = FakeRetriever(
        results=[
            (document, 0.74),
        ]
    )

    contextualizer = FakeContextualizer()
    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
        contextualizer=contextualizer,
    )

    thread_id = "contextualization-test"

    asyncio.run(
        graph.answer(
            query="چطور مصرف آب کشاورزی را کاهش دهیم؟",
            thread_id=thread_id,
        )
    )

    asyncio.run(
        graph.answer(
            query="این روش چه مزیتی دارد؟",
            thread_id=thread_id,
        )
    )

    assert contextualizer.calls == 1

    assert contextualizer.questions[0] == (
        "این روش چه مزیتی دارد؟"
    )

    assert len(contextualizer.histories[0]) == 2

    assert (
        contextualizer.histories[0][0].content
        == "چطور مصرف آب کشاورزی را کاهش دهیم؟"
    )

    assert (
        contextualizer.histories[0][1].content
        == "پاسخ آزمایشی بر اساس منابع بازیابی‌شده"
    )

    assert retriever.calls == 2

    assert retriever.queries[0] == (
        "چطور مصرف آب کشاورزی را کاهش دهیم؟"
    )

    assert retriever.queries[1] == (
        "مزایای آبیاری قطره‌ای "
        "در کاهش مصرف آب کشاورزی چیست؟"
    )
    
def test_rag_graph_limits_short_term_memory():
    retriever = FakeRetriever(
        results=[]
    )

    model = FakeModel()

    contextualizer = FakeContextualizer()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
        contextualizer=contextualizer,
    )

    thread_id = "memory-limit-test"

    for index in range(1, 12):
        asyncio.run(
            graph.answer(
                query=f"پیام {index}",
                thread_id=thread_id,
            )
        )

    snapshot = graph.get_state(
        thread_id
    )

    messages = snapshot.values["messages"]

    assert len(messages) == 20

    assert messages[0].type == "human"
    assert messages[0].content == "پیام 2"

    assert messages[1].type == "ai"

    assert messages[-2].type == "human"
    assert messages[-2].content == "پیام 11"

    assert messages[-1].type == "ai"

    assert contextualizer.calls == 10
    

def test_rag_graph_loads_user_preferences_across_threads():
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()

    memory_store = UserMemoryStore(
        store=store
    )

    memory_store.save_explicit(
        user_id="user-001",
        memory_id="language",
        content="کاربر پاسخ‌های فارسی می‌خواهد.",
    )

    document = Document(
        page_content=(
            "آبیاری قطره‌ای می‌تواند "
            "مصرف آب را کاهش دهد."
        ),
        metadata={
            "source": "agriculture.txt",
            "source_type": "evaluation",
            "chunk_index": 1,
            "chunk_count": 1,
        },
    )

    retriever = FakeRetriever(
        results=[
            (document, 0.72),
        ]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
        store=store,
    )

    first_response = asyncio.run(
        graph.answer(
            query="چطور مصرف آب را کاهش دهیم؟",
            thread_id="thread-1",
            user_id="user-001",
        )
    )

    second_response = asyncio.run(
        graph.answer(
            query="چه روشی مناسب است؟",
            thread_id="thread-2",
            user_id="user-001",
        )
    )

    assert first_response.answer == (
        "پاسخ آزمایشی بر اساس "
        "منابع بازیابی‌شده"
    )

    assert second_response.answer == (
        "پاسخ آزمایشی بر اساس "
        "منابع بازیابی‌شده"
    )

    snapshot = graph.get_state(
        "thread-2"
    )

    assert snapshot.values[
        "user_preferences"
    ][0]["content"] == (
        "کاربر پاسخ‌های فارسی می‌خواهد."
    )
    
def test_normal_conversation_does_not_write_long_term_memory():
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()

    memory_store = UserMemoryStore(
        store=store
    )

    document = Document(
        page_content=(
            "آبیاری قطره‌ای می‌تواند "
            "مصرف آب را کاهش دهد."
        ),
        metadata={
            "source": "agriculture.txt",
            "source_type": "evaluation",
            "chunk_index": 1,
            "chunk_count": 1,
        },
    )

    retriever = FakeRetriever(
        results=[
            (document, 0.72),
        ]
    )

    model = FakeModel()

    rag_chain = RAGChain(
        retriever=retriever,
        model=model,
    )

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=rag_chain,
        store=store,
    )

    asyncio.run(
        graph.answer(
            query=(
                "من پاسخ کوتاه را ترجیح می‌دهم."
            ),
            thread_id="thread-1",
            user_id="user-001",
        )
    )

    assert memory_store.get(
        user_id="user-001",
        memory_id="response-style",
    ) is None