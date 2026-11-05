from fastapi.testclient import TestClient
from app.api.v1.endpoints.rag import get_rag_graph
from app.main import create_app
from app.rag.rag_chain import RAGResponse
import hashlib

class FakeRAGGraph:
    def __init__(self):
        self.calls = []

    async def answer(self, query: str, k: int = 4, score_threshold: float | None = None, thread_id: str = "default", user_id: str | None = None) -> RAGResponse:
        self.calls.append({"query": query, "k": k, "thread_id": thread_id})
        return RAGResponse(answer="The indexed documents contain relevant information.", sources=[])

def create_test_client(fake_graph: FakeRAGGraph) -> TestClient:
    application = create_app(use_in_memory=True)
    application.dependency_overrides[get_rag_graph] = lambda: fake_graph
    return TestClient(application)

def test_rag_endpoint_returns_answer():
    fake_graph = FakeRAGGraph()
    with create_test_client(fake_graph) as client:
        response = client.post(
            "/api/v1/rag/threads/conversation-001/ask",
            json={"question": "  How can water consumption be reduced?  ", "k": 3},
            headers={"X-User-ID": "test_user"}
        )

    assert response.status_code == 200
    
    # Calculate the expected SHA256 hashed thread ID
    expected_raw = "test_user::conversation-001".encode("utf-8")
    expected_hash = hashlib.sha256(expected_raw).hexdigest()

    assert fake_graph.calls == [
        {
            "query": "How can water consumption be reduced?",
            "k": 3,
            "thread_id": expected_hash,
        }
    ]

def test_rag_endpoint_rejects_blank_question():
    fake_graph = FakeRAGGraph()
    with create_test_client(fake_graph) as client:
        response = client.post("/api/v1/rag/threads/conversation-001/ask", json={"question": "   "})
    assert response.status_code == 422

def test_rag_endpoint_rejects_invalid_retrieval_limit():
    fake_graph = FakeRAGGraph()
    with create_test_client(fake_graph) as client:
        response = client.post("/api/v1/rag/threads/conversation-001/ask", json={"question": "Ask a question.", "k": 11})
    assert response.status_code == 422