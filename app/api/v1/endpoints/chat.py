from fastapi import APIRouter, HTTPException, Request, Depends
from langchain_core.messages import HumanMessage
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.models.schemas import ChatRequest, ChatResponse
from app.rag.rag_graph import RAGGraph
from app.dependencies import get_current_user
# 🟢 FIX: Import Guardrail
from app.services.guardrail_service import InputGuardrail, SecurityViolationError

router = APIRouter(prefix="/api/v1/chat", tags=["Chat Interface"])
logger = logging.getLogger(__name__)

_mcp_client = None
_mcp_tools = []

async def get_mcp_tools() -> list:
    global _mcp_client, _mcp_tools
    if _mcp_client is None:
        try:
            logger.info("Initializing MCP Client for API...")
            _mcp_client = MultiServerMCPClient(
                {"local_system": {"command": "python", "args": ["scripts/local_mcp_server.py"], "transport": "stdio"}}
            )
            _mcp_tools = await _mcp_client.get_tools()
        except Exception as e:
            logger.error(f"Failed to initialize MCP Client: {e}")
            _mcp_tools = []
    return _mcp_tools

@router.post("/", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest, fastapi_req: Request, user_id: str = Depends(get_current_user)):
    # 🟢 FIX: Enforce Input Guardrail before invoking the Agent
    guardrail = InputGuardrail()
    try:
        await guardrail.verify_safe_content(request.query)
    except SecurityViolationError as exc:
        raise HTTPException(status_code=400, detail=exc.message)

    try:
        mcp_tools = await get_mcp_tools()
        base_rag_graph: RAGGraph = fastapi_req.app.state.rag_graph
        
        agent = RAGGraph(
            mcp_tools=mcp_tools,
            store=base_rag_graph.store,
            checkpointer=base_rag_graph.checkpointer,
            retriever=base_rag_graph.retriever
        )
        
        config = {"configurable": {"thread_id": request.user_id, "user_id": user_id}}
        result = await agent.graph.ainvoke({"messages": [HumanMessage(content=request.query)]}, config=config)
        final_answer = result["messages"][-1].content
        
        return ChatResponse(answer=final_answer)

    except Exception as e:
        logger.error(f"Error during chat generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing your request.")