
# AI Content Copilot

A local-first AI content and research copilot built with Python, FastAPI, LangChain, LangGraph, Ollama, and Qdrant.

The project provides content processing and document-grounded research using local AI models. Its architecture is designed to remain provider-agnostic while keeping local execution and free tools as the default.

## Project Status

- Stage 1 — Completed: Core content processing API, local Ollama integration, structured outputs, standardized errors, automated tests, and n8n/Telegram integration.
- Stage 2 — Completed: Document ingestion, vector retrieval, RAG, evidence assessment, bounded alternative search, and source-aware abstention.
- Stage 3 — Next: Containerizing the application and improving reproducible deployment.

## Features

### Content Processing

- Text, URL, and PDF inputs
- Summarization, analysis, and content idea generation
- Source-language detection and explicit output-language handling
- Structured AI outputs validated with Pydantic
- Ollama and mock AI providers
- Standardized API errors and transient connection retries
- n8n and Telegram integration

### Document Research and RAG

- Ingest text, URLs, and PDF documents
- Split documents into chunks and preserve source metadata
- Generate embeddings locally using Ollama
- Store and retrieve document chunks using Qdrant
- Answer questions using retrieved document evidence
- Return source information with RAG answers
- Maintain conversation state by thread
- Evaluate whether retrieved evidence supports the user's full question
- Try one alternative retrieval query when evidence is insufficient
- Expand retrieval for the alternative search within a bounded limit
- Abstain when adequate evidence cannot be found after the permitted attempt

The evidence evaluator and alternative-query planner use the configured local language model. Retrieved document contents are treated as untrusted evidence rather than system instructions.

## Architecture

### Content Processing

```text
Text / URL / PDF
        |
        v
Input Normalization
        |
        v
Language Detection
        |
        v
Task Registry
        |
        v
Content Service
        |
        v
AI Provider
   |         |
 Mock      Ollama
        |
        v
Structured Output
```

### Document Research

```text
Text / URL / PDF
        |
        v
Document Ingestion
        |
        v
Chunking and Metadata
        |
        v
Local Embeddings
        |
        v
Qdrant Vector Store
        |
        v
Question and Retrieval
        |
        v
Evidence Sufficiency Evaluation
        |
        +---- Evidence insufficient
        |             |
        |             v
        |    One Alternative Search
        |             |
        |             v
        |     Evaluate Evidence Again
        |
        v
Grounded Answer with Sources
        |
        or
        |
        v
Abstain if Evidence Remains Insufficient
```

PostgreSQL provides persistent application state and LangGraph checkpointing in the normal application configuration. In-memory implementations are used in the test environment.

## API

Interactive API documentation is available at `/docs` while the application is running locally.

### Content Processing

Analyze text or a URL:

```text
POST /api/v1/content/analyze-content
```

Analyze a PDF for content-processing tasks:

```text
POST /api/v1/content/analyze-pdf
```

### Document Ingestion

Ingest supported text or URL content into the document retrieval system:

```text
POST /api/v1/documents/ingest
```

Ingest a PDF document:

```text
POST /api/v1/documents/ingest-pdf
```

### Ask a Research Question

```text
POST /api/v1/rag/threads/{thread_id}/ask
```

Example request:

```json
{
  "question": "What do the indexed documents establish about AI agents?",
  "k": 4
}
```

The `question` field contains the research question. The optional `k` field controls the requested number of retrieved results and defaults to `4`; the supported range is `1` to `10`.

RAG answers include source information from retrieved documents. If the available evidence is inadequate after the bounded alternative search, the system returns an abstention response rather than deliberately generating an unsupported answer.

## Local Components

- **Language model:** Ollama, currently configured with `qwen3:4b`
- **Embedding model:** Ollama, currently configured with `qwen3-embedding:0.6b`
- **Vector database:** Qdrant, collection `content_copilot`
- **Persistent state and checkpointing:** PostgreSQL
- **API framework:** FastAPI
- **RAG orchestration:** LangGraph
- **Data validation:** Pydantic

Model names, service addresses, and credentials are configured through the application's settings and local environment configuration.

## Local Setup

Use Python `3.13` or a compatible version satisfying the project's declared requirements.

Install the project and development dependencies:

```powershell
python -m pip install -e ".[dev]"
```

Create or update the local `.env` configuration using `.env.example` as a reference. Do not commit `.env` or put passwords and other secrets in source control.

Ensure that the configured PostgreSQL, Qdrant, and Ollama services are running and that the configured Ollama models are available.

Run the API:

```powershell
uvicorn app.main:app --reload
```

The application will be available at:

- API: `http://127.0.0.1:8000`
- Interactive documentation: `http://127.0.0.1:8000/docs`

The document ingestion and RAG endpoints require the relevant local services to be available.

## Testing

Run the complete automated test suite:

```powershell
python -m pytest -q
```

The test suite includes coverage for API validation, document ingestion, Qdrant retrieval behavior, RAG graph routing, evidence sufficiency, alternative research queries, bounded retries, and abstention.

Manual evaluation and local integration scripts are available under `scripts/` and `tests/`. These may require the configured local services and models to be running.

The retrieval threshold and ranking results are evaluation-dependent. The current threshold is based on a small local evaluation dataset and is not a guarantee of retrieval quality on arbitrary documents.

## n8n and Telegram

The existing integration can route Telegram messages through n8n to the local FastAPI service. The integration layer supports text messages, task commands, URL inputs, response formatting, and returning results to the originating chat.

The core AI service remains independent of n8n and Telegram.

## Project Structure

```text
ai-content-copilot/
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   ├── memory/
│   ├── models/
│   ├── providers/
│   ├── rag/
│   ├── retrieval/
│   ├── services/
│   └── main.py
├── scripts/
├── tests/
├── .env.example
├── .gitignore
├── compose.yaml
├── LICENSE
├── pyproject.toml
└── README.md
```

## Roadmap

- **Stage 1:** Core AI content-processing service — completed.
- **Stage 2:** Document ingestion, retrieval, RAG, evidence evaluation, bounded alternative research, and source-aware abstention — completed.
- **Stage 3:** Application containerization and reproducible local deployment.
- **Stage 4:** AI security, prompt-injection defenses, data-leakage prevention, guardrails, and logging improvements.
- **Stage 5:** MCP, tool calling, and more advanced agent workflows.
