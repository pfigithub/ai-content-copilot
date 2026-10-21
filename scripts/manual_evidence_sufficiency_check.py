"""Trace live evidence evaluation and bounded alternative retrieval."""

import asyncio

from app.rag.evidence_sufficiency import EvidenceSufficiencyEvaluator
from app.rag.rag_chain import RAGChain
from app.rag.rag_graph import RAGGraph
from app.rag.research_query_planner import ResearchQueryPlanner
from app.retrieval.retriever import ContentRetriever


class TracingRetriever:
    """Record retrieval attempts while using the real Qdrant retriever."""

    def __init__(self, retriever):
        self.retriever = retriever
        self.calls = []

    async def retrieve(self, query, k=4, score_threshold=None):
        results = await self.retriever.retrieve(
            query=query,
            k=k,
            score_threshold=score_threshold,
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


class TracingEvidenceEvaluator:
    """Log real model assessments without changing their decisions."""

    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.calls = []

    async def evaluate(self, question, results):
        assessment = await self.evaluator.evaluate(
            question=question,
            results=results,
        )
        self.calls.append(assessment)

        print(f"\nEvidence assessment {len(self.calls)}:")
        print(f"  Sufficient: {assessment.sufficient}")
        print(f"  Reason: {assessment.reason}")

        return assessment


class TracingQueryPlanner:
    """Log alternative-query planning using the real local model."""

    def __init__(self, planner):
        self.planner = planner
        self.calls = []

    async def create_alternative_query(self, question, attempted_query):
        alternative = await self.planner.create_alternative_query(
            question=question,
            attempted_query=attempted_query,
        )
        self.calls.append(
            {
                "question": question,
                "attempted_query": attempted_query,
                "alternative_query": alternative,
            }
        )

        print("\nAlternative query planning:")
        print(f"  Original query: {attempted_query}")
        print(f"  Alternative query: {alternative}")

        return alternative


async def main() -> int:
    """Exercise evidence evaluation against the existing local documents."""

    question = (
        "برای پیاده‌سازی سامانهٔ مطرح‌شده دربارهٔ پردازش فاکتورها، "
        "آستانهٔ عددی ارجاع فاکتور به بررسی انسانی، الگوریتم دقیق "
        "تشخیص فاکتورهای تکراری و مسیر endpoint مربوط به API حسابداری "
        "را از منابع استخراج کن. این مشخصات باید مستند به اسناد باشند؛ "
        "حدس یا پیشنهاد شخصی ارائه نکن."
    )

    retriever = TracingRetriever(ContentRetriever())
    evaluator = TracingEvidenceEvaluator(
        EvidenceSufficiencyEvaluator()
    )
    planner = TracingQueryPlanner(ResearchQueryPlanner())
    answer_chain = RAGChain(retriever=retriever)

    graph = RAGGraph(
        retriever=retriever,
        rag_chain=answer_chain,
        query_planner=planner,
        evidence_evaluator=evaluator,
    )

    try:
        response = await graph.answer(
            query=question,
            k=8,
            score_threshold=0.4,
            thread_id="manual-evidence-sufficiency-check",
        )
    except Exception as exc:
        print(f"\nFAIL: {type(exc).__name__}: {exc}")
        return 2

    print("\nFinal answer:")
    print(response.answer)

    print("\nFinal sources:")
    for source in response.sources:
        print(
            f"  {source.source} | "
            f"score={source.score}"
        )

    print("\nExperiment summary:")
    print(f"  Retrieval attempts: {len(retriever.calls)}")
    print(f"  Evidence assessments: {len(evaluator.calls)}")
    print(f"  Alternative-query attempts: {len(planner.calls)}")

    if not retriever.calls or not retriever.calls[0]["result_count"]:
        print(
            "\nCHECK: The initial search found no documents, "
            "so this run did not isolate the insufficient-evidence path."
        )
        return 1

    if not evaluator.calls:
        print("\nFAIL: Retrieved evidence was not evaluated.")
        return 1

    if evaluator.calls[0].sufficient:
        print(
            "\nCHECK: The live evaluator considered the initial evidence "
            "sufficient. Review its reason and the final answer."
        )
        return 1

    if len(retriever.calls) == 2 and len(evaluator.calls) == 2:
        if not evaluator.calls[1].sufficient and not response.sources:
            print(
                "\nPASS: Insufficient evidence triggered one alternative "
                "retrieval, and the graph abstained when the second "
                "assessment was also insufficient."
            )
            return 0

        if evaluator.calls[1].sufficient and response.sources:
            print(
                "\nPASS: The alternative retrieval supplied evidence "
                "that the evaluator considered sufficient."
            )
            return 0

    print(
        "\nCHECK: The graph followed a different path than the two "
        "expected insufficient-evidence outcomes. Review the trace."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))