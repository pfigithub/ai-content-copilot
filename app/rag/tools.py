import logging
import json
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_community.tools import DuckDuckGoSearchRun

from app.retrieval.retriever import ContentRetriever
from app.rag.evidence_sufficiency import EvidenceSufficiencyEvaluator
from app.rag.research_query_planner import ResearchQueryPlanner
from app.rag.query_contextualizer import QueryContextualizer
from app.rag.rag_chain import RAGChain

logger = logging.getLogger(__name__)

@tool
async def search_internal_documents(query: str, config: RunnableConfig) -> str:
    """
    Search the internal database (Qdrant) for documents, PDFs, and personal knowledge base.
    USE THIS TOOL ONLY WHEN the user asks about specific facts, uploaded files, or personal knowledge.
    """
    try:
        user_id = config.get("configurable", {}).get("user_id")
        k = config.get("configurable", {}).get("k", 8)
        score_threshold = config.get("configurable", {}).get("score_threshold")
        chat_history = config.get("configurable", {}).get("chat_history", [])
        
        # 1. Instantiate the integrated RAG pipeline components
        retriever = ContentRetriever()
        evaluator = EvidenceSufficiencyEvaluator()
        planner = ResearchQueryPlanner()
        contextualizer = QueryContextualizer()
        rag_chain = RAGChain(retriever=retriever)
        
        # 2. Contextualize the query based on chat history
        try:
            standalone_query = await contextualizer.contextualize(question=query, history=chat_history)
        except Exception as e:
            logger.warning(f"Contextualization failed: {e}. Falling back to raw query.")
            standalone_query = query
            
        # 3. Initial Retrieval (Tenant-isolated)
        results = await retriever.retrieve(query=standalone_query, k=k, score_threshold=score_threshold, user_id=user_id)
        
        if not results:
            return "Insufficient evidence found in the documents to definitively answer this question."
            
        # 4. Evaluator & Planner Loop
        try:
            assessment = await evaluator.evaluate(question=standalone_query, results=results)
            
            if not assessment.sufficient:
                logger.info("Initial evidence insufficient. Attempting alternative query...")
                alt_query = await planner.create_alternative_query(question=standalone_query, attempted_query=standalone_query)
                
                if alt_query and alt_query.strip().casefold() != standalone_query.strip().casefold():
                    alt_results = await retriever.retrieve(query=alt_query, k=k, score_threshold=score_threshold, user_id=user_id)
                    
                    if alt_results:
                        alt_assessment = await evaluator.evaluate(question=standalone_query, results=alt_results)
                        if alt_assessment.sufficient:
                            # Replace with the better alternative results
                            results = alt_results
                        else:
                            return "Insufficient evidence found in the documents to definitively answer this question."
                    else:
                        return "Insufficient evidence found in the documents to definitively answer this question."
                else:
                    return "Insufficient evidence found in the documents to definitively answer this question."
        except Exception as e:
            logger.error(f"RAG Evaluator/Planner encountered an error: {e}. Failing closed to prevent hallucination.")
            return "Unable to verify evidence sufficiency due to an internal evaluation error."
            
        # 5. Generate Grounded Answer via RAGChain (to get structured sources)
        final_rag_response = await rag_chain.answer_from_results(
            query=standalone_query, 
            results=results, 
            history=chat_history
        )
        
        # 6. Format the output so the Agent sees both the grounded answer and the structured sources
        # We inject the JSON string of sources so the graph can parse it later if needed.
        sources_data = [src.model_dump() for src in final_rag_response.sources]
        sources_json = json.dumps(sources_data, ensure_ascii=False)
        
        formatted_response = (
            f"ANSWER:\n{final_rag_response.answer}\n\n"
            f"--- INTERNAL SOURCES DATA ---\n{sources_json}"
        )
            
        return formatted_response
        
    except Exception as e:
        logger.error(f"Internal search critical error: {e}")
        return "The internal database is currently unavailable or an unexpected error occurred."

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