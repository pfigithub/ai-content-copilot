"""
Ultimate test script: Combines static Tools (Qdrant, Web) 
with dynamic MCP Tools (Time, Files) in the LangGraph Agent.
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage
from app.rag.rag_graph import RAGGraph

async def main():
    print("1. Booting up MCP Server to discover dynamic tools...")
    client = MultiServerMCPClient(
        {
            "local_system": {
                "command": "python",
                "args": ["scripts/local_mcp_server.py"],
                "transport": "stdio",
            }
        }
    )
    
    # Extract dynamic tools from the MCP server
    mcp_tools = await client.get_tools()
    
    print("2. Initializing Agent and combining all tools...")
    # Pass the discovered MCP tools into our RAGGraph
    agent = RAGGraph(mcp_tools=mcp_tools)
    
    print(f"Agent's Weaponized Arsenal: {[t.name for t in agent.tools]}\n")
    
    print("--- Ultimate Agent Multi-Tool Test ---")
    query = "ساعت دقیق الان چنده؟ و ضمناً برنامه مدیریت آب شهر چیست؟"
    print(f"User: {query}")
    
    result = await agent.graph.ainvoke({"messages": [HumanMessage(content=query)]})
    print(f"\nAI Final Answer:\n{result['messages'][-1].content}")
    
    print("\n--- Backend Decision Trace ---")
    for msg in result['messages']:
        if msg.type == 'ai' and msg.tool_calls:
            for tool_call in msg.tool_calls:
                print(f"[System Log]: Agent chose tool -> {tool_call['name']}")
                
    if hasattr(client, 'close'):
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())