from app.providers.ai_provider import AIProvider
from app.providers.ollama_provider import OllamaProvider


def get_ai_provider() -> AIProvider:
    return OllamaProvider()