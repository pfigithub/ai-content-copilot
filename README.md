# AI Content Copilot

A local-first, enterprise-ready AI content and research copilot built with Python, FastAPI, LangGraph, Ollama, and Qdrant.

This project provides secure, tenant-isolated content processing and document-grounded research (Agentic RAG) using local AI models, with the ability to fall back or hybridize with cloud models (e.g., Gemini).

## Project Status

- **Stage 1 — Core Processing:** Content analysis API, Ollama integration, structured outputs, and automated tests.
- **Stage 2 — Document Ingestion & RAG:** Secure document ingestion, vector retrieval, contextualized RAG, evidence assessment, and alternative search planning.
- **Stage 3 — Docker Deployment:** Full containerization (API, Frontend, PostgreSQL, Qdrant) with internal networking and health checks via `compose.yaml`.
- **Stage 4 — AI Security (Defense-in-Depth):** Implemented `InputGuardrail`, `OutputGuardrail`, and `IngestionGuardrail` to block Prompt Injection, Data Leakage, and Jailbreaks. Enforced strict SSRF protection for URLs.
- **Stage 5 — Autonomous Agent & MCP:** Tool-Calling Agent integrating internal Qdrant search, Web search, and dynamic local OS tools via Model Context Protocol (MCP) with restricted workspace access.
- **Stage 6 — Enterprise Readiness:** Multi-tenant isolation (user-level data segregation), robust Bearer Token authentication, and persistent Memory Proposals via PostgreSQL.

## Prerequisites

- **Docker & Docker Compose:** Required to run the full stack seamlessly.
- **Ollama:** Installed locally on your host machine to run the AI models.
- **Python 3.13+:** (Optional) If you intend to run the API outside of Docker for development.

## Setup & Execution

### 1. Configure Environment

Copy the example environment file and create your actual `.env` file:
```bash
cp .env.example .env