from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama
from pydantic import BaseModel

from app.config import settings


class ContextualizedQuery(BaseModel):
    query: str


class QueryContextualizer:
    def __init__(
        self,
        model: Runnable | None = None,
    ):
        if model is None:
            model = ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=0,
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You rewrite the user's latest question "
                        "into a self-contained search query.\n\n"
                        "Use the conversation history only to resolve "
                        "references such as 'this method', 'that one', "
                        "'it', or other context-dependent expressions.\n\n"
                        "Preserve the user's original intent.\n"
                        "Do not answer the question.\n"
                        "Return only the rewritten standalone query."
                    ),
                ),
                (
                    "human",
                    (
                        "Conversation history:\n"
                        "{history}\n\n"
                        "Latest question:\n"
                        "{question}"
                    ),
                ),
            ]
        )

        structured_model = model.with_structured_output(
            ContextualizedQuery
        )

        self.chain = prompt | structured_model

    async def contextualize(
        self,
        question: str,
        history: list[BaseMessage] | None = None,
    ) -> str:
        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        history_text = self._format_history(
            history or []
        )

        result = await self.chain.ainvoke(
            {
                "history": history_text,
                "question": question,
            }
        )

        return result.query.strip()

    @staticmethod
    def _format_history(
        messages: list[BaseMessage],
    ) -> str:
        if not messages:
            return "No previous conversation."

        parts = []

        for message in messages:
            if message.type == "human":
                role = "User"
            elif message.type == "ai":
                role = "Assistant"
            else:
                role = message.type

            parts.append(
                f"{role}: {message.content}"
            )

        return "\n".join(parts)