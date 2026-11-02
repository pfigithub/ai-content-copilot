import pytest
import asyncio
from app.rag.rag_graph import RAGGraph

class MockRetriever:
    async def retrieve(self, query, k=4, score_threshold=None):
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
    # اجرای تابع به صورت همگام (Synchronous) برای رفع خطای pytest
    response = asyncio.run(agent.answer("What is the capital of France?", thread_id="test_thread"))
    assert response is not None
    assert isinstance(response.answer, str)
    assert response.answer == "Test response"