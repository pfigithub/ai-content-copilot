import asyncio
from langchain_core.messages import HumanMessage
from app.rag.rag_graph import RAGGraph

async def main():
    agent = RAGGraph()

    print("--- Test 1: Normal Chat ---")
    query1 = "سلام، خسته نباشید. تو کی هستی؟"
    print(f"User: {query1}")
    
    result1 = await agent.graph.ainvoke({"messages": [HumanMessage(content=query1)]})
    print(f"AI: {result1['messages'][-1].content}\n")
    
    print("--- Test 2: Knowledge Query ---")
    query2 = "برنامه مدیریت آب شهر چیست؟"
    print(f"User: {query2}")
    
    result2 = await agent.graph.ainvoke({"messages": [HumanMessage(content=query2)]})
    print(f"AI: {result2['messages'][-1].content}")
    
    # چاپ لاگ ابزارها برای بررسی رفتار Agent
    for msg in result2['messages']:
        if msg.type == 'ai' and msg.tool_calls:
            print(f"\n[System Log]: Agent decided to use tool -> {msg.tool_calls[0]['name']}")

    print("\n--- Test 3: Web Search Query ---")
    query3 = "پایتخت کشور ژاپن کجاست و امروز هوا در آنجا چطور است؟"
    print(f"User: {query3}")
    
    result3 = await agent.graph.ainvoke({"messages": [HumanMessage(content=query3)]})
    print(f"AI: {result3['messages'][-1].content}")
    
    for msg in result3['messages']:
        if msg.type == 'ai' and msg.tool_calls:
            print(f"\n[System Log]: Agent decided to use tool -> {msg.tool_calls[0]['name']}")

if __name__ == "__main__":
    asyncio.run(main())