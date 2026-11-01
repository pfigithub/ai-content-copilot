# AI Content Copilot

A local-first AI content and research copilot built with Python, FastAPI, LangChain, LangGraph, Ollama, and Qdrant.

The project provides content processing and document-grounded research using local AI models. Its architecture is designed to remain provider-agnostic while keeping local execution and free tools as the default.

## Project Status

- Stage 1 — Completed: Core content processing API, local Ollama integration, structured outputs, standardized errors, automated tests, and n8n/Telegram integration.
- Stage 2 — Completed: Document ingestion, vector retrieval, RAG, evidence assessment, bounded alternative search, and source-aware abstention.
- Stage 3 — Completed: Application containerization, Docker networking, and reproducible local deployment via `compose.yaml`.
- Stage 4 — Completed: AI Security (Defense-in-Depth). Implemented `InputGuardrail`, `OutputGuardrail`, and `IngestionGuardrail` to defend against Prompt Injection, Data Leakage, and Jailbreaks.
- Stage 5 — Completed: Autonomous Agent Architecture & MCP. Refactored the core graph to a Tool-Calling Agent, integrated internal Qdrant search, DuckDuckGo web search, and dynamic local OS tools via the Model Context Protocol (MCP).

## Features

### Content Processing
- Text, URL, and PDF inputs
- Summarization, analysis, and content idea generation
- Source-language detection and explicit output-language handling
- Structured AI outputs validated with Pydantic

### Autonomous RAG & Tool Calling (Agentic AI)
- Autonomous LangGraph Agent that routes queries intelligently.
- Seamlessly switches between static tools (Vector DB, Web Search) and dynamic tools discovered via MCP (Local Time, File System).
- Robust 3-layer security system scanning inputs, outputs, and ingested documents.
- Maintains conversation state by thread using PostgreSQL checkpointers.

## API Endpoints

Interactive API documentation is available at `/docs` while the application is running.

- **POST /api/v1/content/analyze-content**: Analyze text or URLs.
- **POST /api/v1/documents/ingest**: Vectorize and store documents securely.
- **POST /api/v1/chat/**: Interact seamlessly with the autonomous Agent (Accesses Web, Qdrant, and MCP).

## Local Setup

Use Python `3.13` and set up the virtual environment:
```powershell
python -m pip install -e ".[dev]"