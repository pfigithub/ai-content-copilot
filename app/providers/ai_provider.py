from abc import ABC, abstractmethod


class AIProvider(ABC):

    @abstractmethod
    async def analyze_content(
        self,
        content: str,
        task: str | None = None,
        instruction: str | None = None,
        source_language: str | None = None,
        output_language: str | None = None,
    ) -> dict:
        """Process content with an AI implementation."""
        raise NotImplementedError