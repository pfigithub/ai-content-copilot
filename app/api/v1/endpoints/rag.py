from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Request,
)

from app.models.schemas import RAGQueryRequest
from app.rag.rag_chain import RAGResponse
from app.rag.rag_graph import RAGGraph


router = APIRouter(
    prefix="/api/v1/rag",
    tags=["rag"],
)


def get_rag_graph(request: Request) -> RAGGraph:
    """Retrieve the shared RAG graph initialized by the application lifespan."""

    return request.app.state.rag_graph


@router.post(
    "/threads/{thread_id}/ask",
    response_model=RAGResponse,
)
async def ask_rag_question(
    thread_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=128,
            description="Identifier of the conversation thread.",
        ),
    ],
    request: RAGQueryRequest,
    rag_graph: Annotated[
        RAGGraph,
        Depends(get_rag_graph),
    ],
) -> RAGResponse:
    """Answer a question using retrieved documents and thread history."""

    normalized_thread_id = thread_id.strip()

    if not normalized_thread_id:
        raise HTTPException(
            status_code=422,
            detail="Thread ID cannot contain only whitespace.",
        )

    # Reuse the application's shared graph and its configured checkpointer.
    return await rag_graph.answer(
        query=request.question,
        k=request.k,
        thread_id=normalized_thread_id,
    )