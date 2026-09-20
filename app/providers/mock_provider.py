from app.providers.ai_provider import AIProvider


class MockAIProvider(AIProvider):

    async def analyze_content(
        self,
        content: str,
        task: str | None = None,
    ) -> dict:
        return {
            "title": "Content analysis result",
            "summary": f"Mock analysis: {content}",
            "importance": 50,
            "category": "other",
        }