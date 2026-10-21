from fastapi.testclient import TestClient

from app.api.v1.endpoints.rag import get_rag_graph
from app.main import create_app
from app.rag.rag_chain import RAGResponse


class FakeRAGGraph:
    """Simulate RAG responses without calling Ollama or Qdrant."""

    def __init__(self):
        """Initialize a list for recording graph calls."""

        self.calls = []

    async def answer(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
    ) -> RAGResponse:
        """Record the request and return a predictable test response."""

        self.calls.append(
            {
                "query": query,
                "k": k,
                "thread_id": thread_id,
            }
        )

        return RAGResponse(
            answer="The indexed documents contain relevant information.",
            sources=[],
        )


def create_test_client(
    fake_graph: FakeRAGGraph,
) -> TestClient:
    """Create a test client with the RAG graph dependency replaced."""

    application = create_app(
        use_in_memory=True,
    )

    application.dependency_overrides[get_rag_graph] = (
        lambda: fake_graph
    )

    return TestClient(application)


def test_rag_endpoint_returns_answer():
    """Verify that the endpoint forwards the question and thread ID."""

    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={
                "question": "  How can water consumption be reduced?  ",
                "k": 3,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["answer"] == (
        "The indexed documents contain relevant information."
    )

    assert body["sources"] == []

    assert fake_graph.calls == [
        {
            "query": "How can water consumption be reduced?",
            "k": 3,
            "thread_id": "conversation-001",
        }
    ]


def test_rag_endpoint_rejects_blank_question():
    """Verify that a whitespace-only question is rejected."""

    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={
                "question": "   ",
            },
        )

    assert response.status_code == 422
    assert fake_graph.calls == []


def test_rag_endpoint_rejects_unknown_fields():
    """Verify that the request schema rejects unsupported fields."""

    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={
                "question": "Ask a question.",
                "unexpected": True,
            },
        )

    assert response.status_code == 422
    assert fake_graph.calls == []


def test_rag_endpoint_rejects_invalid_retrieval_limit():
    """Verify that k stays within the configured API limits."""

    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={
                "question": "Ask a question.",
                "k": 11,
            },
        )

    assert response.status_code == 422
    assert fake_graph.calls == []