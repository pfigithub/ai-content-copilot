import asyncio

from langchain_core.runnables import RunnableLambda

from app.rag.research_query_planner import ResearchQueryPlanner


class FakeStructuredModel:
    """Simulate structured model output without calling Ollama."""

    def __init__(self, query):
        self.query = query
        self.messages = None

    def with_structured_output(self, schema):
        async def invoke(messages):
            self.messages = messages
            return schema(query=self.query)

        return RunnableLambda(invoke)


def test_planner_returns_a_distinct_alternative_query():
    """Verify the planner returns a different search query."""

    model = FakeStructuredModel(
        "اثر سایهٔ درختان و تبخیر آب برگ‌ها بر دمای شهر"
    )
    planner = ResearchQueryPlanner(model=model)

    result = asyncio.run(
        planner.create_alternative_query(
            question="درختان چگونه شهر را خنک می‌کنند؟",
            attempted_query="درختان چگونه شهر را خنک می‌کنند؟",
        )
    )

    assert result == (
        "اثر سایهٔ درختان و تبخیر آب برگ‌ها بر دمای شهر"
    )
    assert len(model.messages) == 2
    assert "درختان چگونه شهر را خنک می‌کنند؟" in (
        model.messages[1].content
    )


def test_planner_returns_none_when_query_is_unchanged():
    """Verify the planner rejects an identical alternative query."""

    original_query = "How do trees cool cities?"
    model = FakeStructuredModel(original_query)
    planner = ResearchQueryPlanner(model=model)

    result = asyncio.run(
        planner.create_alternative_query(
            question=original_query,
            attempted_query=original_query,
        )
    )

    assert result is None