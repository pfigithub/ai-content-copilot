"""
Test script to demonstrate how an AI Agent discovers and uses tools 
from an external MCP Server dynamically.
"""

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from app.config import settings

async def main():
    print("1. Connecting to Local MCP Server via stdio...")
    
    # 1. Instantiate the client with the required 'transport' key
    client = MultiServerMCPClient(
        {
            "local_system": {
                "command": "python",
                "args": ["scripts/local_mcp_server.py"],
                "transport": "stdio",  # <--- این خط برای حل ارور اضافه شد
            }
        }
    )
    
    print("2. Server connected! Extracting available tools...")
    
    # 2. Extract tools using await
    tools = await client.get_tools()
    
    # Look at the magic! The tools are discovered dynamically.
    print(f"Discovered Tools: {[tool.name for tool in tools]}\n")
    
    # 3. Bind the discovered tools to our local LLM
    model = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )
    model_with_tools = model.bind_tools(tools)
    
    print("3. Testing the Agent with discovered MCP tools...")
    query = "ساعت و تاریخ فعلی سیستم را بگو و لیست فایل‌های پوشه فعلی را نشون بده."
    print(f"User: {query}")
    
    # 4. Ask the model to decide what to do
    response = await model_with_tools.ainvoke([HumanMessage(content=query)])
    
    print(f"\nAI Decision (Tool Calls):")
    for tool_call in response.tool_calls:
        print(f"- Decided to use tool: {tool_call['name']} with arguments: {tool_call['args']}")
        
    # Clean up connections
    if hasattr(client, 'close'):
        await client.close()

if __name__ == "__main__":
    asyncio.run(main())