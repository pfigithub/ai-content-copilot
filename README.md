# AI Content Copilot

A local-first AI content and research copilot built with Python, FastAPI, and Ollama.

The project focuses on practical AI backend engineering, local LLM integration, structured AI processing, automation, and a provider-agnostic architecture.

## Current Status

Stage 1 — Completed

Stage 1 provides a working local AI content processing service with text, URL, and PDF inputs, task-aware processing, multilingual handling, standardized errors, Ollama integration, automated tests, and n8n + Telegram integration.

## Features

* Text, URL, and PDF input
* Summarize, analyze, and ideas tasks
* Source-language detection
* Explicit output-language handling
* Local Ollama integration
* Mock AI provider for testing
* Structured AI outputs with Pydantic validation
* Source-grounded processing and prompt-injection boundaries
* Standardized API errors
* Transient Ollama connection retries
* n8n and Telegram integration

## Architecture

```text
Input
├── Text
├── URL
└── PDF
      ↓
Input Normalization
      ↓
Language Detection
      ↓
Task Registry
      ↓
Content Service
      ↓
AI Provider
├── MockAIProvider
└── OllamaProvider
      ↓
Structured Output
```

Integration:

```text
Telegram
   ↓
n8n
   ↓
FastAPI
   ↓
Ollama
   ↓
n8n
   ↓
Telegram
```

The core AI service remains independent from n8n and Telegram.

## Supported Tasks

### Summarize

Produces a grounded summary based on the provided source.

### Analyze

Analyzes the main claims, reasoning, evidence, weaknesses, and implications supported by the source.

### Ideas

Generates several distinct and practical content ideas grounded in the source.

## Supported Inputs

* Text
* URL
* PDF

## API

### Analyze Content

```text
POST /api/v1/content/analyze-content
```

Example request:

```json
{
  "text": "Artificial intelligence is changing content creation.",
  "task": "summarize"
}
```

### Analyze PDF

```text
POST /api/v1/content/analyze-pdf
```

## AI Provider

The application uses a provider abstraction so the core service is not tied to a single AI implementation.

Current providers:

* MockAIProvider
* OllamaProvider

Default local model:

```text
qwen3:4b
```

## Local Setup

Create and activate a virtual environment, then install the project:

```powershell
python -m pip install -e ".[dev]"
```

Make sure Ollama is running and the configured model is available.

Run the API:

```powershell
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Testing

Run the complete test suite:

```powershell
python -m pytest -q
```

## Error Handling

The API uses a standardized error contract.

| Code                       | HTTP |
| -------------------------- | ---: |
| VALIDATION_ERROR           |  422 |
| PDF_EXTRACTION_FAILED      |  422 |
| URL_FETCH_FAILED           |  502 |
| AI_PROVIDER_RESPONSE_ERROR |  502 |
| AI_PROVIDER_UNAVAILABLE    |  503 |
| AI_PROVIDER_TIMEOUT        |  504 |

## n8n + Telegram

n8n can send requests to the local FastAPI service through:

```text
http://host.docker.internal:8000
```

The integration layer supports Telegram text messages, task commands, URL inputs, response formatting, and returning results to the originating chat.

## Project Structure

```text
ai-content-copilot/
├── app/
│   ├── api/
│   ├── models/
│   ├── providers/
│   ├── services/
│   └── task_registry.py
├── tests/
├── .env.example
├── .gitignore
├── LICENSE
├── pyproject.toml
└── README.md
```

## Stage 1 Result

Stage 1 establishes the core AI backend and local integration foundation for the next stage of the project.

## Next Stage

Stage 2 will focus on professional agent capabilities:

* LangChain
* LangGraph
* RAG
* Vector Database
