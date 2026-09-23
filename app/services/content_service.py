from app.models.schemas import (
    AnalysisContentResponse,
    IdeasContentResponse,
    NormalizedContent,
    SummaryContentResponse,
)

from app.providers.ai_provider import AIProvider
from app.task_registry import get_task_definition
from app.services.output_language_service import (
    resolve_output_language,
)


async def analyze_content(
    content: NormalizedContent,
    ai_provider: AIProvider,
) -> (
    SummaryContentResponse
    | AnalysisContentResponse
    | IdeasContentResponse
):

    task_definition = get_task_definition(
        content.task
    )
    
    output_language = (
        resolve_output_language(
            content.instruction,
            content.source_language,
        )
    )

    ai_result = await ai_provider.analyze_content(
        content=content.content,
        task=content.task,
        instruction=content.instruction,
        source_language=content.source_language,
        output_language=output_language,
    )   

    common_data = {
        "input_type": content.source_type,
        "title": ai_result["title"],
        "category": ai_result["category"],
    }

    if task_definition.name in (
        "summarize",
        "default",
    ):

        return SummaryContentResponse(
            **common_data,
            summary=ai_result["summary"],
            importance=ai_result["importance"],
        )

    if task_definition.name == "analyze":

        return AnalysisContentResponse(
            **common_data,
            analysis=ai_result["analysis"],
            importance=ai_result["importance"],
        )

    if task_definition.name == "ideas":

        return IdeasContentResponse(
            **common_data,
            summary=ai_result["summary"],
            ideas=ai_result["ideas"],
        )

    raise ValueError(
        "Unsupported task."
    )