from app.providers.ai_provider import AIProvider
from app.task_registry import get_task_definition


class MockAIProvider(AIProvider):

    async def analyze_content(
        self,
        content: str,
        task: str | None = None,
        instruction: str | None = None,
        source_language: str | None = None,
        output_language: str | None = None,
    ) -> dict:

        task_definition = get_task_definition(
            task
        )

        if task == "analyze":

            result = {
                "title": "Content analysis result",
                "analysis": (
                    f"Mock analysis: {content}"
                ),
                "importance": 50,
                "category": "other",
            }

        elif task == "ideas":

            result = {
                "title": "Content ideas",
                "summary": (
                    f"Mock idea direction: {content}"
                ),
                "ideas": [
                    "Idea one",
                    "Idea two",
                    "Idea three",
                    "Idea four",
                    "Idea five",
                ],
                "category": "other",
            }

        else:

            result = {
                "title": "Content analysis result",
                "summary": (
                    f"Mock analysis: {content}"
                ),
                "importance": 50,
                "category": "other",
            }

        validated_result = (
            task_definition.result_model
            .model_validate(result)
        )

        return validated_result.model_dump()