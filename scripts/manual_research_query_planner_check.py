"""Manually verify alternative query planning with the local Ollama model."""

import asyncio

from app.rag.research_query_planner import ResearchQueryPlanner


async def main() -> int:
    """Run one live query-planning check."""

    question = (
        "این خبر جدید درباره استفاده از عامل‌های هوشمند برای "
        "خودکارسازی کارها چه ارتباطی با یادداشت‌های قبلی من "
        "درباره عامل‌های هوش مصنوعی دارد؟"
    )

    attempted_query = (
        "خبر جدید درباره عامل‌های هوشمند و خودکارسازی کارها"
    )

    planner = ResearchQueryPlanner()

    print("Original question:")
    print(question)

    print("\nInitial retrieval query:")
    print(attempted_query)

    try:
        alternative_query = await planner.create_alternative_query(
            question=question,
            attempted_query=attempted_query,
        )
    except Exception as exc:
        print("\nFAIL: Local query planning raised an exception.")
        print(f"{type(exc).__name__}: {exc}")
        return 2

    print("\nAlternative retrieval query:")
    print(alternative_query or "<No distinct alternative produced>")

    if not alternative_query:
        print("\nFAIL: The planner did not produce a distinct query.")
        return 1

    if alternative_query.casefold() == attempted_query.casefold():
        print("\nFAIL: The alternative query is unchanged.")
        return 1

    print("\nPASS: A distinct alternative query was produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))