import logging
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun
from app.retrieval.retriever import ContentRetriever
from app.rag.evidence_sufficiency import EvidenceSufficiencyEvaluator
from app.rag.research_query_planner import ResearchQueryPlanner

logger = logging.getLogger(__name__)

@tool
async def search_internal_documents(query: str, config: RunnableConfig) -> str:
    """
    Search the internal database (Qdrant) for documents, PDFs, and personal knowledge base.
    USE THIS TOOL ONLY WHEN the user asks about specific facts, uploaded files, or personal knowledge.
    """
    try:
        user_id = config.get("configurable", {}).get("user_id")
        retriever = ContentRetriever()
        evaluator = EvidenceSufficiencyEvaluator()
        planner = ResearchQueryPlanner()
        
        # 1. Initial Retrieval
        results = await retriever.retrieve(query=query, k=8, user_id=user_id)
        
        if not results:
            return "No relevant information found in the internal database."
            
        # 2. Self-Reflective Evaluation
        try:
            assessment = await evaluator.evaluate(question=query, results=results)
            if not assessment.sufficient:
                logger.info(f"Evidence insufficient. Planning alternative...")
                alt_query = await planner.create_alternative_query(question=query, attempted_query=query)
                
                if alt_query and alt_query.strip().casefold() != query.strip().casefold():
                    alt_results = await retriever.retrieve(query=alt_query, k=8, user_id=user_id)
                    
                    # Merge results (avoiding duplicates by source and chunk_index)
                    seen = {(d.metadata.get("source"), d.metadata.get("chunk_index")) for d, _ in results}
                    for doc, score in alt_results:
                        key = (doc.metadata.get("source"), doc.metadata.get("chunk_index"))
                        if key not in seen:
                            results.append((doc, score))
                            seen.add(key)
        except Exception as e:
            logger.warning(f"Advanced RAG evaluation skipped due to error: {e}")
            
        # 4. Format and Return Final Context
        context_parts = []
        for index, (document, _) in enumerate(results, start=1):
            source = document.metadata.get("source", "unknown")
            source_type = document.metadata.get("source_type", "unknown")
            
            page_meta = document.metadata.get("page")
            page_num = str(int(page_meta) + 1) if isinstance(page_meta, (int, float)) or (isinstance(page_meta, str) and page_meta.isdigit()) else "unknown"
            
            context_parts.append(
                f"<document index=\"{index}\" source=\"{source}\" type=\"{source_type}\" page=\"{page_num}\">\n"
                f"{document.page_content}\n"
                f"</document>"
            )
            
        return "\n\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"Internal search error: {e}")
        return "The internal database is currently empty or unavailable. Tell the user you don't have internal documents right now."

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