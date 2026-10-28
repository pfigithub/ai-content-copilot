from langchain_core.tools import tool
from app.retrieval.retriever import ContentRetriever
from langchain_community.tools import DuckDuckGoSearchRun

@tool
async def search_internal_documents(query: str) -> str:
    """
    Search the internal database (Qdrant) for documents, PDFs, and personal knowledge base.
    USE THIS TOOL ONLY WHEN the user asks about specific facts, uploaded files, or personal knowledge.
    """
    retriever = ContentRetriever()
    results = await retriever.retrieve(query=query, k=4)
    
    if not results:
        return "No relevant information found in the internal database."
        
    context_parts = []
    for index, (document, _) in enumerate(results, start=1):
        source = document.metadata.get("source", "unknown")
        source_type = document.metadata.get("source_type", "unknown")
        
        # XML Encapsulation for safety
        context_parts.append(
            f"<document index=\"{index}\" source=\"{source}\" type=\"{source_type}\">\n"
            f"{document.page_content}\n"
            f"</document>"
        )
        
    return "\n\n".join(context_parts)

@tool
def search_web(query: str) -> str:
    """
    Search the internet for current events, weather, news, or general knowledge.
    USE THIS TOOL ONLY WHEN you need up-to-date real-world information.
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.invoke(query)
        if not result or result.strip() == "":
            return "Web search returned no results."
        return result
    except Exception as e:
        return f"Web search failed due to an error: {str(e)}"