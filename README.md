# Agent Orchestration System

A local React + FastAPI workbench for testing an OmniRoute-powered LangGraph agent.
Celery runs agent jobs, Redis stores graph checkpoints, and PostgreSQL/pgvector
provides hybrid document search.

## What the agent is supposed to do

The agent follows a plan → act → reflect workflow. Its registered tools are
limited to:

- **Search internal documents:** keyword and semantic retrieval over documents
  you explicitly index.
- **Delete an internal document:** requires your approval in the React app
  before the deletion tool runs.

It can also answer general questions through the configured OmniRoute model. It
cannot browse the web, read files from your computer, or run shell commands.
The app displays the generated plan, task status, any pending approval, and the
final answer.

## Start the app on Windows

Prerequisites: Docker Desktop, Node.js (npm included), Python 3.11+, and an
OmniRoute API key with access to an OpenAI-compatible chat-completions API.
Docker is only the convenient local way this project starts Redis Stack and
PostgreSQL; React and FastAPI do not require Docker if you provide compatible
services yourself and configure `REDIS_URL` and `DATABASE_URL` in `.env`.

1. If this is the first run, create the private environment file:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   Set `LLM_API_KEY` to your OmniRoute key. `LLM_BASE_URL` is
   `http://localhost:20128/v1`. Verify `LLM_MODEL` matches an enabled model ID in
   your OmniRoute account. Keep the key only in `.env`, never in source code or
   chat.

2. From the project folder run:

   ```powershell
   .\Start-AgentLab.ps1
   ```

   The launcher starts Redis Stack and PostgreSQL with Docker Compose, installs
   the Python and React dependencies if needed, starts the Celery worker and
   FastAPI, then starts Vite. Open **http://127.0.0.1:5173** while the launcher
   is running. The React dev server proxies API requests to FastAPI on port 8000.

   If PowerShell blocks the script, use this for the current terminal only:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\Start-AgentLab.ps1
   ```

### Test a full agent run

1. Confirm the top of the app says **API online** and the model key is
   configured. If not, check `.env` and restart the launcher.
2. Enter a general question and click **Run agent**. When the request completes,
   the run trace shows the plan and model response.
3. To test retrieval, use **Documents & hybrid search** to index a small,
   non-sensitive sample document. Search for a phrase in that document.
4. Ask the agent to search for that same topic. The agent should call its
   internal search tool and summarize the retrieved text.
5. To test human approval, index a disposable sample with ID `demo-policy`, then
   ask: `Delete the internal document with ID demo-policy.` Review the proposed
   tool call and click **Approve action** or **Reject action**. Rejection must
   not delete the document.

Do not use sensitive data for these first tests. A successful UI load does not
guarantee the model provider is available; if OmniRoute returns an error, check
the provider dashboard and the Celery worker window.

## Local database and embeddings

You do **not** need to open PostgreSQL or create a database manually. Compose
creates a local `agent_orchestration` database, document table, pgvector
extension, and Redis Stack service. Redis Stack is required because LangGraph's
Redis checkpointer uses Redis Search commands such as `FT.INFO`; plain Redis
does not provide them. PostgreSQL and Redis data persist in Docker volumes. Do
not use `docker compose down -v` unless you intend to erase that local data.

If you started the project before Redis Stack was configured and see
`unknown command 'FT.INFO'`, replace the existing Redis container from the
project folder:

```powershell
docker compose pull redis
docker compose up -d --force-recreate redis
```

This recreates only the Redis container; it does not remove named data volumes.
If you still have checkpoint data in an older Redis container that had no
persistent volume, that container's data may not survive recreation. Restart
`.\Start-AgentLab.ps1` afterward. The app health status checks Redis, its Search
module, and the Celery worker separately.

The first indexing/search request downloads `BAAI/bge-small-en-v1.5`
automatically through FastEmbed. It produces the 384-dimensional vectors the
database expects; no manual model download is needed.

## Configuration and API

`.env.example` has placeholders only. The project reads `LLM_API_KEY`,
`LLM_BASE_URL`, `LLM_MODEL`, `REDIS_URL`, `DATABASE_URL`, and `EMBEDDING_MODEL`.
This project uses OmniRoute; it does not require an Anthropic key. An
`APPROVAL_API_KEY` is optional for loopback development, and required for
approval requests from non-local clients.

FastAPI docs are at `http://127.0.0.1:8000/docs`. Keep the API and React
development server on loopback; do not expose this development setup to a
network without adding appropriate authentication and authorization.
