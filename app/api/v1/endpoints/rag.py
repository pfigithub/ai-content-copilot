from typing import Annotated
import hashlib
from fastapi import APIRouter, Depends, HTTPException, Path, Request
from app.models.schemas import RAGQueryRequest
from app.rag.rag_chain import RAGResponse
from app.rag.rag_graph import RAGGraph
from app.dependencies import get_current_user

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

def get_rag_graph(request: Request) -> RAGGraph:
    return request.app.state.rag_graph

@router.post("/threads/{thread_id}/ask", response_model=RAGResponse)
async def ask_rag_question(
    thread_id: Annotated[str, Path(min_length=1, max_length=128, description="Identifier of the conversation thread.")],
    request: RAGQueryRequest,
    rag_graph: Annotated[RAGGraph, Depends(get_rag_graph)],
    user_id: str = Depends(get_current_user),
) -> RAGResponse:
    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise HTTPException(status_code=422, detail="Thread ID cannot contain only whitespace.")

    # Creating an unambiguous, deterministic thread ID
    raw_identity = f"{user_id}::{normalized_thread_id}".encode("utf-8")
    secure_thread_id = hashlib.sha256(raw_identity).hexdigest()

    return await rag_graph.answer(
        query=request.question,
        k=request.k,
        thread_id=secure_thread_id,
        user_id=user_id
    )