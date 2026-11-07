"""Trace the live RAG graph while exercising alternative retrieval."""

import asyncio
from app.rag.rag_graph import RAGGraph

class TracingRetriever:
    """Log real retrieval calls while delegating to the existing retriever."""
    def __init__(self, retriever):
        self.retriever = retriever
        self.calls = []

    async def retrieve(self, query, k=4, score_threshold=None, user_id=None):
        results = await self.retriever.retrieve(
            query=query,
            k=k,
            score_threshold=score_threshold,
            user_id=user_id,
        )

        record = {
            "query": query,
            "k": k,
            "score_threshold": score_threshold,
            "result_count": len(results),
            "sources": [
                document.metadata.get("source", "unknown")
                for document, _score in results
            ],
            "scores": [
                round(float(score), 5)
                for _document, score in results
            ],
        }
        self.calls.append(record)

        print(f"\nRetrieval attempt {len(self.calls)}:")
        print(f"  Query: {query}")
        print(f"  k: {k}")
        print(f"  Threshold: {score_threshold}")
        print(f"  Results: {record['result_count']}")
        print(f"  Sources: {record['sources']}")
        print(f"  Scores: {record['scores']}")

        return results

async def main() -> int:
    """Run one bounded live retrieval experiment."""
    from app.retrieval.retriever import ContentRetriever
    
    question = (
        "با مقایسه‌ی یادداشت قبلی درباره‌ی عامل‌های هوشمند و "
        "خبر آزمایشی درباره‌ی خودکارسازی پردازش فاکتورها، "
        "ارتباط این دو سند چیست؟"
    )

    retriever = TracingRetriever(ContentRetriever())
    
    # We pass only the tracing retriever; the graph's tools will instantiate the rest internally
    graph = RAGGraph(retriever=retriever)

    try:
        response = await graph.answer(
            query=question,
            k=8,
            score_threshold=0.6,
            thread_id="manual-agent-retry-check",
            user_id="manual_test_user"
        )
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 2

    print("\nFinal answer:")
    print(response.answer)

    print("\nFinal sources:")
    for source in response.sources:
        print(f"  {source.source} | score={source.score}")

    if len(retriever.calls) == 2:
        first_query = retriever.calls[0]["query"]
        second_query = retriever.calls[1]["query"]

        if first_query != second_query:
            print("\nPASS: The graph performed one alternative retrieval with a different query.")
            return 0

    print("\nCHECK: The graph did not complete the expected two-query alternative-retrieval path.")
    return 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))