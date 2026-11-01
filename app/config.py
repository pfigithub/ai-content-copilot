from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "content_copilot"
    gemini_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()