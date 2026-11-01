# Agent Platform

A small platform to build AI agents on your own documents, started with one command:

- **Module A: knowledge bases.** Upload documents (txt, md, pdf). They are chunked,
  embedded and indexed in PostgreSQL with pgvector, and semantic search can be tested
  from the UI.
- **Module B: agents.** An agent has instructions, a model, tools (calculator, weather,
  web search) and linked knowledge bases. It decides at each step whether to call a
  tool or to answer directly, and every step is saved and shown.
- **Module C: interface.** A React app to manage knowledge bases, configure agents
  (Agent Builder) and chat with them (Playground), with the agent's reasoning shown
  live, step by step.

## Quick start

Requirements: Docker with Docker Compose v2.

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 (interactive docs: http://localhost:8000/docs) |
| PostgreSQL + pgvector | localhost:5432 (local only) |

Database migrations run automatically when the backend starts (`alembic upgrade head`),
after PostgreSQL passes its healthcheck. The first indexing downloads the embedding
model (about 100 MB, kept in a Docker volume across rebuilds).

**No API key needed to try it.** With `LLM_API_KEY` empty, the platform runs in
**mock mode**: a rule-based fake model that calls tools and answers, clearly labelled
as a mock in the UI. Everything else (knowledge bases, search, tools, traces) is real.

### A five-minute tour

1. **Knowledge bases**: create one, upload a PDF, watch its status go from *pending* to
   *ready*, then try the search tester (top chunks with their cosine scores).
2. **Agents → Create an agent**: name it, keep or edit the instructions, pick a model,
   tick *calculator* and *weather*, link the knowledge base.
3. **Playground**: ask *"What's the weather in Lyon, and what is 15% of 2,340?"*. The
   trace shows each LLM call, each tool call with its arguments and result, then the
   answer. Ask a follow-up question: the agent remembers the conversation.

## Configuration

All settings come from `.env` (documented line by line in `.env.example`). Never commit
`.env`: it holds API keys.

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | `agent`, `agent`, `agent_platform` | Database credentials |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible endpoint (OpenRouter, Ollama, vLLM, a company gateway) |
| `LLM_API_KEY` | empty | Empty = mock mode |
| `LLM_MODEL` | `openrouter/free` | Default model; each agent can choose its own |
| `OLLAMA_BASE_URL` | empty | Optional: add a local Ollama as a second provider |
| `OPENROUTER_API_KEY` | empty | Optional: add OpenRouter as a provider, when the default is another one |
| `TAVILY_API_KEY` | empty | Optional: enables the web search tool (free plan) |
| `VITE_API_URL` | `http://localhost:8000` | API URL as seen by the browser (built into the frontend) |
| `CORS_ORIGINS` | `http://localhost:5173` | Origins allowed to call the API |

**Using a real model.**
- **OpenRouter:** create a free key at https://openrouter.ai/keys and set `LLM_API_KEY`.
  `openrouter/free` routes to a free model that supports tool calling (free models:
  50 requests a day without credits).
- **Local model with Ollama:** set `LLM_BASE_URL=http://host.docker.internal:11434/v1`,
  `LLM_API_KEY=ollama` and a model that supports tool calling, e.g. `qwen2.5:7b`. On
  Linux, the backend also needs `extra_hosts: ["host.docker.internal:host-gateway"]`.

When several providers are configured, the Agent Builder lists every provider's models
live. Models that cannot call tools are greyed out, since an agent needs them.

## Architecture

```
 Browser ── React (Vite, TypeScript) ─────────────────────────────────────────┐
   │  Knowledge bases · Agent Builder · Playground (live trace over SSE)     │
   ▼                                                                         │
 FastAPI ── routes (HTTP, validation) ── services (logic) ── db (SQLAlchemy) │
   │            │                            │                               │
   │            │      ┌─────────────────────┼──────────────────────┐        │
   │            │      │ indexing: extract → chunk → embed (FastEmbed, CPU)  │
   │            │      │ agent loop: LLM ⇄ tools, every step saved          │
   │            │      │ LLM layer: OpenAI format, mock, several providers   │
   │            │      └─────────────────────┬──────────────────────┘        │
   ▼            ▼                            ▼                               │
 PostgreSQL 16 + pgvector: knowledge bases, documents, chunks (HNSW index),  │
 agents, conversations, runs, run steps  ◄── Alembic migrations at boot ─────┘
```

```
backend/
  app/api/routes/      HTTP only: parse, validate, map errors to status codes
  app/services/        the logic: chunking, embeddings, search, agent loop, tools, LLM
  app/db/              SQLAlchemy models and session
  app/schemas/         Pydantic request and response models
  alembic/versions/    migrations (applied at startup)
  eval/                benchmarks: routing and retrieval (see EVALUATION.md)
  tests/               pytest
frontend/src/
  api/                 typed client, SSE parser
  hooks/               data loading, live run stream
  components/, pages/  UI
```

### How a question is answered

1. The browser sends the question to `POST /agents/{id}/runs/stream`.
2. The backend creates the run and starts the agent loop in a worker thread with its
   own database session. Steps go through a queue and are sent to the browser as
   Server-Sent Events as soon as they happen.
3. The loop sends the system prompt (the agent's instructions plus the platform's
   rules), the conversation history and the question, with the schema of every tool.
4. The model either answers, or asks for tools. Tools run and their results go back
   to the model. A failing tool becomes an error message the model can react to.
5. The loop stops on an answer, or after the agent's maximum number of LLM calls.
   Each step is committed as it happens, so a run that fails halfway still shows how
   far it got.

## Choices and trade-offs

Organised by the evaluation criteria.

### Orchestration and RAG

- **Chunking in tokens, with each embedding model's own tokenizer** (default 500
  tokens, overlap 50). Counting words or characters would overflow or underuse the
  model's input. The maximum chunk size depends on the model, and the settings are
  frozen per knowledge base: changing them would mix incompatible chunks.
- **Three CPU embedding models** (FastEmbed, ONNX): multilingual e5-small by default,
  because documents and questions may be in French and English. No GPU or external
  API needed. Query and passage prefixes (e5) are handled inside the embedding service.
- **pgvector with an HNSW index on cosine distance**, in the same database as
  everything else: one service to run, transactional consistency between documents
  and chunks. HNSW over IVFFlat: no training step, good recall on a growing table.
  Trade-off: a dedicated vector database scales further.
- **Indexing in a background task**, with a status per document (pending, processing,
  ready, error) polled by the UI, and recovery of interrupted documents at startup.
  Trade-off: FastAPI background tasks, not a job queue; fine for one instance.
- **Routing (tool use vs direct answer) is left to the model**, guided by precise tool
  descriptions and platform rules. A linked knowledge base gives the agent a search
  tool whose description includes the knowledge bases' names and descriptions, so the
  model knows when it is relevant. Measured in `EVALUATION.md`.
- **Tool results are treated as untrusted data**: the platform's rules tell the model
  that instructions found in documents or web pages are not instructions to follow
  (prompt injection).

### Software architecture

- **Routes, services, database are separate layers.** Routes only translate HTTP;
  the agent loop depends on a small `ChatModel` interface, not on a provider SDK, so
  tests drive it with a scripted fake model.
- **One client for every provider**: OpenRouter, Ollama, vLLM and company gateways all
  speak the OpenAI format; only the URL and key change.
- **Errors**:
  - tool failures go back to the model as observations;
  - LLM provider errors (bad key, rate limit, model without tool support) become a
    clear message on the run;
  - platform bugs are logged and shown as a generic error, never disguised as a
    provider problem;
  - 404 / 422 are checked before a stream starts.
  The OpenAI SDK retries 429 and 5xx with backoff.
- **Every run is an audit trail**: it keeps a snapshot of the model, provider, system
  prompt and tools used, since the agent may be edited later.
- **Synchronous SQLAlchemy** in a thread pool: simpler and enough here. Async would
  matter with many concurrent long runs.
- **Frontend without a UI library**: React and TypeScript only, typed API client,
  small hooks. Fewer dependencies to explain and to update.

### User experience

- **Agent Builder**:
  - tools that cannot work (no web search key) are shown as unavailable, with the
    reason;
  - linking a knowledge base shows that it adds the search tool;
  - the model dropdown lists each provider's models and greys out those without tool
    calling.
- **Playground**:
  - conversations with history (the last 10 question-and-answer turns are resent);
  - a Stop button;
  - each step appears live: LLM calls with token counts, tool calls with arguments,
    result and duration, sources (document and rank, or web page).
- **Mock mode is visible**: a banner says answers are not from a real model.

### DevOps and execution

- `docker compose up --build` starts PostgreSQL (`pgvector/pgvector:pg16`, with
  healthcheck), the backend on port 8000 and the frontend on port 5173 (nginx).
- Migrations run at boot, before the API starts. Embedding weights are cached in a
  volume.
- **Starts without any account** (mock mode), so the project can be reviewed offline.
- Optional public demo behind Cloudflare Tunnel and Access (email one-time code),
  with no open port: see `DEPLOY_PUBLIC.md`.

## Evaluation

Two benchmarks, matching the criteria "precision of the pgvector search" and
"relevance of agentic routing". Datasets, method, results and limits are in
[`EVALUATION.md`](EVALUATION.md).

- **Routing:**
  - 21 questions with the expected decision (which tool, or a direct answer), traps
    included;
  - measures accuracy, over-calls, under-calls and consistency across repeats.
- **Retrieval:**
  - 26 questions (10 in French) on a fictional fund documentation set with look-alike
    documents;
  - measures Recall@1/3/5 and MRR for each embedding model and chunk size, through the
    real API.

## Tests

```bash
cd backend && uv run pytest        # unit and API tests (database tests skipped without one)
cd frontend && node --test src/api/sse.test.ts   # Node 22.18+ (runs TypeScript directly)
```

## Known limits and next steps

- **Single instance.** Background indexing, run streaming and interrupted-run recovery
  assume one backend process. Next: a job queue (Celery, Arq) and a shared event bus
  for streams.
- **No authentication or multi-tenancy.** Next: users, roles, a knowledge base access
  policy checked at search time.
- **Search is vector only.** Next: hybrid search (PostgreSQL full-text + vectors) and a
  reranker, which help with exact terms such as fund codes.
- **Conversation memory resends the last 10 turns.** Long threads cost more per
  question. Next: summarise older turns, or retrieve only the relevant ones.
- **Answer quality is not evaluated yet**, only routing and retrieval. Next: reference
  answers and an LLM judge for faithfulness to the sources, checked against human
  labels, run in CI on every prompt or model change.
- **Secrets in `.env`.** Next: a secret manager, and per-provider quotas and fallback.
