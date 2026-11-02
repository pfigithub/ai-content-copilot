import pytest
import asyncio
from app.rag.rag_graph import RAGGraph

class MockRetriever:
    async def retrieve(self, query, k=4, score_threshold=None, user_id=None):
        return []

@pytest.fixture
def agent():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.store.memory import InMemoryStore
    store = InMemoryStore()
    checkpointer = InMemorySaver()
    retriever = MockRetriever()
    return RAGGraph(store=store, checkpointer=checkpointer, retriever=retriever)

def test_agent_graph_initializes(agent):
    assert agent.graph is not None
    assert len(agent.tools) >= 2

def test_agent_graph_answers_queries(agent):
    # Setup event loop to avoid pytest-asyncio strict requirement
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        response = loop.run_until_complete(agent.answer("What is the capital of France?", thread_id="test_thread"))
        assert response is not None
        assert isinstance(response.answer, str)
        assert response.answer == "Test response"
    finally:
        loop.close()