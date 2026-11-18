"""Constants shared across the Streamlit frontend."""

import os


API_BASE = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

# Endpoints — centralised here so path changes require one edit.
AUTH_LOGIN = f"{API_BASE}/api/v1/auth/login"
AUTH_REGISTER = f"{API_BASE}/api/v1/auth/register"
AUTH_ME = f"{API_BASE}/api/v1/auth/me"
AUTH_REFRESH = f"{API_BASE}/api/v1/auth/refresh"
AUTH_LOGOUT = f"{API_BASE}/api/v1/auth/logout"
AUTH_ADMIN_USERS = f"{API_BASE}/api/v1/auth/admin/users"

CHAT_URL = f"{API_BASE}/api/v1/chat/"
CONTENT_ANALYZE = f"{API_BASE}/api/v1/content/analyze-content"
CONTENT_PDF = f"{API_BASE}/api/v1/content/analyze-pdf"
INGEST_PDF = f"{API_BASE}/api/v1/documents/ingest-pdf"

# (connect, read) timeout. Read is long because chat can wait on Ollama.
REQUEST_TIMEOUT = (5, 120)

# Auto-logout after this many seconds of inactivity.
INACTIVITY_TIMEOUT_SECONDS = 30 * 60