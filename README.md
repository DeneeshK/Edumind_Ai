# EduMind AI Backend

FastAPI backend for EduMind's adaptive learning product. The service creates
personalized courses, generates module lessons with LLMs, evaluates learner
answers, stores progress in PostgreSQL, and exposes frontend-ready course APIs.

This repository is backend-only. The frontend is deployed separately and talks
to this API over authenticated HTTP and Server-Sent Events.

## Live vs Legacy Code Paths

- The deployed frontend uses **only** the course-centric `/api/courses` flow:
  `app/course_api.py` → `core/course_service.py` → `agents/curriculum_architect.py`
  + `agents/evaluation_agent.py`, with web-search retrieval via
  `clients/mcp_search_client.py` (the standalone `edumind_mcp_search` server).
- A second interactive `/session/*` flow (`app/main.py`, `agents/orchestrator.py`,
  `agents/tutor.py`, `agents/evaluator.py`, `agents/adaptation_engine.py`) is
  **legacy** — kept only as a reference implementation, tagged `legacy-session`
  in `/docs`, and carrying `LEGACY` docstring headers.
- The old in-process ChromaDB/BGE-embedding/reranker retrieval stack has been
  removed; it was disabled in production and never served the live flow.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full breakdown.

## Runtime Components

- FastAPI app: `app/api.py`
- Frontend course API: `app/course_api.py`
- Google OAuth and session cookies: `app/auth.py`
- Agent pipeline: `agents/`
- Course, roadmap, lesson, and validation services: `core/`
- PostgreSQL repository and schema bootstrap: `db/postgres.py`
- External LLM/search clients: `clients/`
- Evaluation metrics and report generation: `evaluation/`
- Docker Compose runtime: `docker-compose.yml`

## Current Production Shape

- Backend domain: `https://course-api.edumindai.org`
- Frontend domain: `https://edumindai.org`
- EC2 repository path: `/home/ubuntu/Edumind_Ai`
- Docker Compose services: `edumind-backend`, `edumind-postgres`
- Nginx terminates SSL and proxies the backend domain to `127.0.0.1:8000`
- The production `.env` file lives only on EC2 and must not be committed

## Quick Start

```bash
cp .env.example .env
```

Fill in local development values for `GROQ_API_KEY`, `TAVILY_API_KEY`,
`DATABASE_URL`, OAuth settings if needed, and session secret values.

Run with Docker Compose:

```bash
docker compose up --build
```

Run locally with an existing Python environment:

```bash
venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

Useful URLs:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Readiness: `http://localhost:8000/ready`
- Metrics: `http://localhost:8000/metrics`

## Tests

```bash
venv/bin/pytest -q
venv/bin/pytest tests/unit -q
venv/bin/pytest tests/integration -q
```

The current test suite uses mocked LLM/auth flows for the lightweight integration
tests. It should not require real Groq, Google, or Tavily network calls.

## Documentation Index

- [Architecture](docs/ARCHITECTURE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Environment Variables](docs/ENVIRONMENT.md)
- [Setup](docs/SETUP.md)
- [Testing](docs/TESTING.md)
- [Logging](docs/LOGGING.md)
- [Developer Handover](docs/DEVELOPER_HANDOVER.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Docker and EC2 Deployment](DOCKER.md)

## Safety Rules

- Do not commit `.env`, API keys, OAuth secrets, private keys, or database URLs
  containing real credentials.
- Do not run `docker compose down -v` in production; it deletes named volumes and
  can remove Postgres data.
- Do not log full prompts, generated lessons, tokens, authorization headers, raw
  documents, or private user content.
- Keep route paths, response shapes, database schema behavior, and LLM prompt
  contracts stable unless the frontend and tests are updated together.
