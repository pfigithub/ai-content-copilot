
import re

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class Settings(BaseSettings):
    """Centralized, validated application settings."""

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    ollama_embedding_model: str = "nomic-embed-text"

    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_collection: str = "content_copilot"

    gemini_api_key: str | None = None

    # Maps SHA-256 API-key digests to server-controlled user IDs.
    api_key_hashes: dict[str, str] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    @field_validator("api_key_hashes")
    @classmethod
    def validate_api_key_hashes(
        cls,
        value: dict[str, str],
    ) -> dict[str, str]:
        """Reject malformed API-key digests and unsafe user identifiers."""

        for digest, user_id in value.items():
            if not _SHA256_PATTERN.fullmatch(digest):
                raise ValueError(
                    "Every API key must be configured as a lowercase SHA-256 digest."
                )

            if not _USER_ID_PATTERN.fullmatch(user_id):
                raise ValueError(
                    "API key user IDs must contain 1-64 letters, digits, "
                    "underscores, or hyphens and must start with a letter or digit."
                )

        return value


settings = Settings()
