# Agent Orchestration Workbench

[![CI](https://github.com/HabibZouari/Agent-Orchestration-Workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/HabibZouari/Agent-Orchestration-Workbench/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A human-in-the-loop AI agent built with **LangGraph**, **FastAPI**, and **React**.
The agent plans, calls tools, and **pauses for human approval before any destructive
action**. It answers questions over an internal knowledge base using **hybrid
retrieval** (keyword + semantic search fused with Reciprocal Rank Fusion).

The same code runs in two modes, chosen in `.env`:

| | **Local mode** (default) | **Distributed mode** |
|---|---|---|
| Task execution | Thread pool inside the API | **Celery** workers over **Redis** |
| Agent checkpoints | In memory | **Redis Stack** (survive restarts, shared by workers) |
| Document store | **SQLite**: FTS5 BM25 + NumPy cosine | **PostgreSQL**: full-text search + **pgvector** HNSW |
| Needs | Python + Node | + Docker (`docker compose up -d`) |

## How it works

```mermaid
flowchart LR
    UI[React workbench] -->|REST| API[FastAPI]
    API -->|submit / poll| EX{{TaskExecutor}}
    EX -->|local| TP[Thread pool]
    EX -->|celery| Q[(Redis broker)] --> W[Celery worker]
    TP & W --> G[LangGraph agent]
    G <-->|checkpoints| CP[(Memory / Redis Stack)]
    G -->|tools| S{{DocumentStore}}
    API -->|index / search| S
    S --> SQ[(SQLite + FTS5)]
    S --> PG[(PostgreSQL + pgvector)]
    G -->|chat completions| LLM[OpenAI-compatible LLM]
```

The agent is a LangGraph state machine:

```mermaid
flowchart LR
    START --> plan --> act
    act -->|no tool calls| reflect
    act -->|tool calls| gate[approval gate]
    gate -->|low risk, or approved| tools --> reflect
    gate -->|rejected| denied[deny tool calls] --> reflect
    reflect --> END
```

1. **Plan**: the model writes a short, structured plan (validated with Pydantic).
2. **Act**: the model chooses tools. Each tool has a declared risk level, and any
   tool without one is treated as high risk.
3. **Approval gate**: for high-risk calls, the graph `interrupt()`s. Its state is
   checkpointed, and the API returns the exact tool call (name and arguments) for
   review. Approving or rejecting resumes the same thread, possibly on a different
   worker.
4. **Reflect**: the model summarises what happened, including denied actions.

The agent's tools are `search_internal_documents` (low risk) and
`delete_internal_document` (high risk). It has no shell, file-system, or web access.

### Hybrid retrieval

Each query runs two retrievers. One is keyword search (SQLite FTS5 BM25 with Porter
stemming, or PostgreSQL `ts_rank_cd`). The other is semantic search over
`BAAI/bge-small-en-v1.5` embeddings (FastEmbed, run locally on CPU). Their rankings
are merged with Reciprocal Rank Fusion, `score = Σ 1 / (60 + rank)`, so scores are
comparable across backends without tuning weights.

## Quick start

**Prerequisites:** Python 3.11+, Node.js 20+, and an API key for any
OpenAI-compatible chat-completions endpoint (OpenAI, OmniRoute, vLLM, Ollama, ...).
Docker is only needed for distributed mode.

```powershell
Copy-Item .env.example .env    # then set LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
.\Start-AgentLab.ps1           # Windows: installs deps, starts API + UI
```

Open **http://127.0.0.1:5173**. The launcher reads `TASK_BACKEND` and `VECTOR_STORE`
from `.env`. In distributed mode it also starts Redis Stack and PostgreSQL with
Docker Compose and opens a Celery worker window. If port 8000 is busy, run
`.\Start-AgentLab.ps1 -ApiPort 8010`.

<details>
<summary>Manual start (macOS / Linux / any shell)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env                       # set LLM_API_KEY etc.

uvicorn agent_orchestration.api:app --port 8000          # terminal 1
cd frontend && npm ci && npm run dev                     # terminal 2

# Distributed mode: set TASK_BACKEND=celery and VECTOR_STORE=pgvector in .env, then
docker compose up -d
celery -A agent_orchestration.celery_app worker --loglevel=info   # terminal 3
```
</details>

### Try it

Three sample documents are indexed on first start: `demo-policy`,
`security-guidelines`, and `agent-playbook`.

1. **Search**: *"Search our internal documents for the data-retention policy and
   summarize it."* The agent calls the search tool without asking.
2. **Approval**: *"Delete the internal document with ID demo-policy."* The run pauses
   and shows the proposed tool call. **Reject** leaves the document in place, and
   **Approve** deletes it. The document list updates either way.
3. **Retrieval only**: index your own text in the RAG sandbox and search it directly.

## API

Interactive docs are served at `http://127.0.0.1:8000/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Mode, model, and the state of Redis, RediSearch, and the worker |
| `POST` | `/agents/run` | Queue a request: `{request, thread_id?}` → `202 {task_id, thread_id}` |
| `GET` | `/tasks/{task_id}` | `PENDING` / `STARTED` / `SUCCESS` (`result`) / `FAILURE` (`error`) |
| `POST` | `/agents/{thread_id}/approval` | Resume a paused run: `{approved: bool}` |
| `GET` | `/documents` | List indexed documents |
| `PUT` | `/documents` | Upsert up to 100 documents: `{documents: [{id, content, metadata}]}` |
| `POST` | `/search` | Hybrid search: `{query, top_k}` |

A paused run returns `result.status = "awaiting_approval"` and `result.interrupts`
with the tool calls to review. Approving a thread that is not paused fails safely.
Runs on the same thread are serialized (with a Redis lock in distributed mode), so a
double-clicked approval cannot execute an action twice.

## Configuration

All settings come from environment variables or `.env`. See [.env.example](.env.example).

| Variable | Default | Notes |
|---|---|---|
| `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` | none / `localhost:20128/v1` / `gemma-2-2b` | Any OpenAI-compatible endpoint |
| `TASK_BACKEND` | `local` | `local` or `celery` |
| `REDIS_URL` | `redis://localhost:6379/0` | Must use db `0`, because RediSearch only indexes db 0 |
| `VECTOR_STORE` | `sqlite` | `sqlite` or `pgvector` |
| `SQLITE_PATH` / `DATABASE_URL` | `data/documents.db` / local Compose Postgres | |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `BAAI/bge-small-en-v1.5` / `384` | Downloaded once on first use |
| `SEED_SAMPLE_DOCUMENTS` | `true` | Index the three samples when the store is empty |
| `APPROVAL_API_KEY` | empty | If set, approvals need the `X-Approval-Token` header. If empty, only loopback clients can approve |

## Development

```bash
pip install -e ".[dev]"
pytest -m "not integration"        # fast, offline: fake LLM + hashing embedder
ruff check src tests && ruff format --check src tests

# Integration tests against real Redis Stack, a Celery worker, and pgvector:
docker compose up -d
TEST_DATABASE_URL=postgresql://agent:agent@localhost:5432/agent_orchestration \
TEST_REDIS_URL=redis://localhost:6379/0 pytest -m integration
```

The unit tests drive the real LangGraph workflow with a scripted chat model. They
cover the full approve and reject paths, double-approval protection, follow-up
requests on a thread, hybrid ranking, and the API's authorization rules. CI runs
lint, unit tests, and integration tests (on Python 3.11 and 3.12), plus the
frontend build.

```
src/agent_orchestration/
├── api.py          FastAPI routes; executor and store injected as dependencies
├── graph.py        LangGraph workflow, checkpointer selection, run/resume
├── executors.py    LocalExecutor (thread pool) and CeleryExecutor behind one interface
├── celery_app.py   Celery app and tasks (with per-thread Redis lock)
├── retrieval.py    SQLiteStore, PgVectorStore, and Reciprocal Rank Fusion
├── tools.py        Agent tools and their risk levels
├── schemas.py      Request and response models
└── settings.py     Typed configuration (pydantic-settings)
frontend/           React + Vite workbench
tests/              Unit tests (offline) and integration tests (live services)
```

## Security notes

This is a local development workbench. The API, UI, and Compose services bind to
`127.0.0.1`. Approvals are restricted to loopback clients unless `APPROVAL_API_KEY`
is set. Secrets stay in `.env`, which git ignores, and `/health` reports only
whether a key is configured. Do not expose this setup to a network without adding
real authentication.

## License

[MIT](LICENSE)
