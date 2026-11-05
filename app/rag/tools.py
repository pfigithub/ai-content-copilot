import logging
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
        # History is passed down from the agent state to give Contextualizer the full picture
        chat_history = config.get("configurable", {}).get("chat_history", [])
        
        retriever = ContentRetriever()
        evaluator = EvidenceSufficiencyEvaluator()
        planner = ResearchQueryPlanner()
        contextualizer = QueryContextualizer()
        rag_chain = RAGChain(retriever=retriever)
        
        try:
            standalone_query = await contextualizer.contextualize(question=query, history=chat_history)
        except Exception as e:
            logger.warning(f"Contextualization failed: {e}. Using raw query.")
            standalone_query = query
            
        results = await retriever.retrieve(query=standalone_query, k=k, score_threshold=score_threshold, user_id=user_id)
        
        if not results:
            return "No relevant information found in the internal database."
            
        try:
            assessment = await evaluator.evaluate(question=standalone_query, results=results)
            
            if not assessment.sufficient:
                logger.info("Evidence insufficient. Planning alternative query...")
                alt_query = await planner.create_alternative_query(question=standalone_query, attempted_query=standalone_query)
                
                if alt_query and alt_query.strip().casefold() != standalone_query.strip().casefold():
                    alt_results = await retriever.retrieve(query=alt_query, k=k, score_threshold=score_threshold, user_id=user_id)
                    
                    if alt_results:
                        alt_assessment = await evaluator.evaluate(question=standalone_query, results=alt_results)
                        if alt_assessment.sufficient:
                            results = alt_results
                        else:
                            return "Insufficient evidence found in the documents to definitively answer this question."
                    else:
                        return "Insufficient evidence found in the documents to definitively answer this question."
                else:
                    return "Insufficient evidence found in the documents to definitively answer this question."
                    
        except Exception as e:
            logger.error(f"RAG Evaluator/Planner failed: {e}. Failing closed to prevent hallucination.")
            return "Unable to answer due to an internal evidence evaluation error."
            
        # user_id is now explicitly passed to the RAGChain to enforce tenant isolation down the line
        final_rag_response = await rag_chain.answer_from_results(query=standalone_query, results=results, history=chat_history)
        
        formatted_response = f"{final_rag_response.answer}\n\nSources Used:\n"
        for src in final_rag_response.sources:
            # Fixing the chunk_index vs page confusion
            page_info = f", Page {int(src.chunk_index) if getattr(src, 'chunk_index', None) is not None else 'unknown'}"
            formatted_response += f"- {src.source} (Score: {src.score:.2f}{page_info})\n"
            
        return formatted_response
        
    except Exception as e:
        logger.error(f"Internal search critical error: {e}")
        return "The internal database is currently empty or unavailable."

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