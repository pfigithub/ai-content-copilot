import hashlib
import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints.rag import get_rag_graph
from app.config import settings
from app.main import create_app
from app.rag.rag_chain import RAGResponse, RAGSource

TEST_TOKEN = "rag-api-test-token"
TEST_USER_ID = "test_user"

def auth_headers() -> dict[str, str]:
    """Build the Bearer authentication header for API tests."""
    return {
        "Authorization": f"Bearer {TEST_TOKEN}",
    }

class FakeRAGGraph:
    """Replace the real RAG graph with a predictable test double."""
    def __init__(self):
        self.calls = []

    async def answer(
        self,
        query: str,
        k: int = 4,
        score_threshold: float | None = None,
        thread_id: str = "default",
        user_id: str | None = None,
    ) -> RAGResponse:
        """Record the API arguments and return a deterministic answer."""
        self.calls.append({
            "query": query,
            "k": k,
            "thread_id": thread_id,
            "user_id": user_id,
        })

        return RAGResponse(
            answer="The indexed documents contain relevant information.",
            sources=[
                RAGSource(source="test_doc.pdf", score=0.95, chunk_index=1, chunk_count=5)
            ],
        )

def create_test_client(
    fake_graph: FakeRAGGraph,
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Create an authenticated test client with an injected fake RAG graph."""
    token_digest = hashlib.sha256(TEST_TOKEN.encode("utf-8")).hexdigest()

    monkeypatch.setattr(
        settings,
        "api_key_hashes",
        {token_digest: TEST_USER_ID},
    )

    application = create_app(use_in_memory=True)
    application.dependency_overrides[get_rag_graph] = lambda: fake_graph
    return TestClient(application)

def test_rag_endpoint_returns_answer(monkeypatch):
    """Return an answer and scope the thread to the authenticated user."""
    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph, monkeypatch) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={
                "question": "  How can water consumption be reduced?  ",
                "k": 3,
            },
            headers=auth_headers(),
        )

    assert response.status_code == 200

    expected_raw = f"{TEST_USER_ID}::conversation-001".encode("utf-8")
    expected_hash = hashlib.sha256(expected_raw).hexdigest()

    assert fake_graph.calls == [
        {
            "query": "How can water consumption be reduced?",
            "k": 3,
            "thread_id": expected_hash,
            "user_id": TEST_USER_ID,
        }
    ]

    assert response.json()["answer"] == "The indexed documents contain relevant information."
    assert len(response.json()["sources"]) == 1
    assert response.json()["sources"][0]["source"] == "test_doc.pdf"

def test_rag_endpoint_rejects_blank_question(monkeypatch):
    """Validate the question after successful authentication."""
    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph, monkeypatch) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={"question": "   "},
            headers=auth_headers(),
        )

    assert response.status_code == 422
    assert fake_graph.calls == []

def test_rag_endpoint_rejects_invalid_retrieval_limit(monkeypatch):
    """Reject retrieval limits outside the request schema constraints."""
    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph, monkeypatch) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={"question": "Ask a question.", "k": 11},
            headers=auth_headers(),
        )

    assert response.status_code == 422
    assert fake_graph.calls == []

def test_rag_endpoint_rejects_legacy_user_id_header(monkeypatch):
    """Do not accept the removed X-User-ID authentication mechanism."""
    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph, monkeypatch) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={"question": "Ask a question."},
            headers={"X-User-ID": TEST_USER_ID},
        )

    assert response.status_code == 401
    assert fake_graph.calls == []

def test_rag_endpoint_requires_authentication(monkeypatch):
    """Reject requests that do not provide authentication credentials."""
    fake_graph = FakeRAGGraph()

    with create_test_client(fake_graph, monkeypatch) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={"question": "Ask a question."},
        )

    assert response.status_code == 401
    assert fake_graph.calls == []