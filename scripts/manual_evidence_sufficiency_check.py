"""Trace live evidence evaluation and bounded alternative retrieval."""

import asyncio
from app.rag.rag_graph import RAGGraph

class TracingRetriever:
    """Record retrieval attempts while using the real Qdrant retriever."""
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
        print(f"  Query: {record['query']}")
        print(f"  k: {record['k']}")
        print(f"  Threshold: {record['score_threshold']}")
        print(f"  Results: {record['result_count']}")
        print(f"  Sources: {record['sources']}")
        print(f"  Scores: {record['scores']}")

        return results


async def main() -> int:
    """Exercise evidence evaluation against the existing local documents."""
    from app.retrieval.retriever import ContentRetriever
    
    question = (
        "برای پیاده‌سازی سامانه‌ی مطرح‌شده درباره‌ی پردازش فاکتورها، "
        "آستانه‌ی عددی ارجاع فاکتور به بررسی انسانی، الگوریتم دقیق "
        "تشخیص فاکتورهای تکراری و مسیر endpoint مربوط به API حسابداری "
        "را از منابع استخراج کن. این مشخصات باید مستند به اسناد باشند؛ "
        "حدس یا پیشنهاد شخصی ارائه نکن."
    )

    retriever = TracingRetriever(ContentRetriever())
    graph = RAGGraph(retriever=retriever)

    try:
        response = await graph.answer(
            query=question,
            k=8,
            score_threshold=0.4,
            thread_id="manual-evidence-sufficiency-check",
            user_id="manual_test_user",
        )
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 2

    print("\nFinal answer:")
    print(response.answer)

    print("\nFinal sources:")
    for source in response.sources:
        print(f"  {source.source} | score={source.score}")

    print("\nExperiment summary:")
    print(f"  Retrieval attempts: {len(retriever.calls)}")

    if not retriever.calls or not retriever.calls[0]["result_count"]:
        print("\nCHECK: The initial search found no documents, so this run did not isolate the insufficient-evidence path.")
        return 1

    if len(retriever.calls) == 2:
        print("\nPASS: Insufficient evidence triggered one alternative retrieval.")
        return 0

    print("\nCHECK: The graph followed a different path than the expected insufficient-evidence outcomes. Review the trace.")
    return 1

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))