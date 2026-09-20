from app.providers.ai_provider import AIProvider
from app.providers.mock_provider import MockAIProvider


def get_ai_provider() -> AIProvider:
    return MockAIProvider()