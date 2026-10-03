from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_ollama import ChatOllama

from app.config import settings
from app.models.schemas import SummaryResult


class SummaryChain:

    def __init__(self):

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a content summarization assistant. "
                        "Summarize the provided source faithfully. "
                        "Do not add facts that are not present in the source."
                    ),
                ),
                (
                    "human",
                    "{content}",
                ),
            ]
        )

        model = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )

        structured_model = (
            model.with_structured_output(
                SummaryResult
            )
        )

        self.chain = (
            prompt
            | structured_model
        )

    async def summarize(
        self,
        content: str,
    ) -> SummaryResult:

        return await self.chain.ainvoke(
            {
                "content": content,
            }
        )