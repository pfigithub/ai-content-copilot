
import os
from pathlib import Path
from urllib.parse import quote

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def get_postgres_uri() -> str:
    file_values = dotenv_values(ENV_FILE)

    # Environment variables override values from .env.
    values = {
        **file_values,
        **os.environ,
    }

    host = values.get("POSTGRES_HOST") or "127.0.0.1"
    port = values.get("POSTGRES_PORT") or "5433"
    database = values.get("POSTGRES_DB") or "ai_content_copilot"

    # Use the dedicated application account, not POSTGRES_USER
    # (which identifies the administrator in the Compose setup).
    username = (
        values.get("POSTGRES_APP_USER")
        or "ai_content_app"
    )

    password = values.get("POSTGRES_APP_PASSWORD")

    if not isinstance(password, str) or not password:
        raise RuntimeError(
            "POSTGRES_APP_PASSWORD is missing "
            "from the environment configuration."
        )

    if username.strip().lower() == "postgres":
        raise RuntimeError(
            "POSTGRES_APP_USER must identify the dedicated "
            "application account, not 'postgres'."
        )

    username = username.strip()

    return (
        f"postgresql://{quote(username, safe='')}:"
        f"{quote(password, safe='')}@"
        f"{host}:{port}/{quote(database, safe='')}"
    )
