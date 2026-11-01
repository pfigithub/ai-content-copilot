from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import HumanMessage
import logging
from langchain_mcp_adapters.client import MultiServerMCPClient

from app.models.schemas import ChatRequest, ChatResponse
from app.rag.rag_graph import RAGGraph

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chat Interface"]
)
logger = logging.getLogger(__name__)

# Global variables to maintain MCP connection across API requests
_mcp_client = None
_mcp_tools = []

async def get_mcp_tools() -> list:
    """Helper function to initialize and fetch dynamic MCP tools."""
    global _mcp_client, _mcp_tools
    
    if _mcp_client is None:
        try:
            logger.info("Initializing MCP Client for API...")
            _mcp_client = MultiServerMCPClient(
                {
                    "local_system": {
                        "command": "python",
                        "args": ["scripts/local_mcp_server.py"],
                        "transport": "stdio",
                    }
                }
            )
            _mcp_tools = await _mcp_client.get_tools()
        except Exception as e:
            logger.error(f"Failed to initialize MCP Client: {e}")
            _mcp_tools = [] # Fallback to static tools if MCP fails

    return _mcp_tools


@router.post("/", response_model=ChatResponse)
async def chat_with_agent(request: ChatRequest, fastapi_req: Request):
    """
    Send a message to the AI Agent. The Agent will decide whether to use tools
    (like Qdrant or Web Search) or answer directly.
    """
    try:
        mcp_tools = await get_mcp_tools()
        base_rag_graph: RAGGraph = fastapi_req.app.state.rag_graph
        
        # We instantiate a NEW graph instance for this request 
        # so the ToolNode compiles correctly with both static and dynamic MCP tools.
        agent = RAGGraph(
            mcp_tools=mcp_tools,
            store=base_rag_graph.store,
            checkpointer=base_rag_graph.checkpointer,
            retriever=base_rag_graph.retriever
        )
        
        config = {"configurable": {"thread_id": request.user_id}}
        
        result = await agent.graph.ainvoke(
            {"messages": [HumanMessage(content=request.query)]},
            config=config
        )
        
        final_answer = result["messages"][-1].content
        
        return ChatResponse(answer=final_answer)

    except Exception as e:
        logger.error(f"Error during chat generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error while processing your request.")