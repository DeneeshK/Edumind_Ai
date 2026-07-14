# Architecture

EduMind is a FastAPI backend organized around a course-centric adaptive learning
flow. The app accepts a learner goal, builds a roadmap, generates lessons,
evaluates answers, adapts future content, and persists state in PostgreSQL.

## Top-Level Folders

| Folder | Role |
| --- | --- |
| `app/` | FastAPI application, HTTP routes, auth, SSE streaming, request/response models. |
| `agents/` | LLM-driven agents for curriculum planning, tutoring, evaluation, adaptation, and reports. |
| `core/` | Service layer for course creation, lesson generation, roadmap conversion, validation, metrics, and student state models. |
| `db/` | PostgreSQL connection pool, schema creation, and repository functions. |
| `clients/` | External provider clients: Groq (LLM), Tavily (YouTube video lookup), and the MCP web-search client that reaches the standalone `edumind_mcp_search` server for retrieval. |
| `evaluation/` | Offline and runtime metrics, evaluation reports, scheduler, and evaluation API endpoints. |
| `tests/` | Unit and mocked integration tests for service behavior, routes, auth, config, and report writing. |
| `monitoring/` | Prometheus and Grafana configuration for local/production observability. |

## Live vs Legacy Code Paths

The deployed frontend uses **only** the course-centric `/api/courses` flow.
A second, older interactive `/session/*` flow (CLI + SSE) is kept as a working
reference implementation but serves no production traffic. Do not confuse them.

| Concern | Live (`/api/courses`) — serves production | Legacy (`/session/*`) — reference only |
| --- | --- | --- |
| HTTP surface | `app/course_api.py`, `app/institution_api.py`, routers mounted in `app/api.py` | `/session/*` endpoints in `app/api.py` (tagged `legacy-session` in `/docs`) |
| Entry point | `uvicorn app.api:app` (web) | `python -m app.main` (CLI) |
| Orchestration | `core/course_service.py` | `agents/orchestrator.py` |
| Curriculum | `agents/curriculum_architect.py` | `agents/curriculum_architect.py` (shared) |
| Tutoring | `core/course_service.py` lesson generation | `agents/tutor.py` |
| Evaluation | `agents/evaluation_agent.py` | `agents/evaluator.py` |
| Adaptation | inline in the course flow | `agents/adaptation_engine.py` |
| Retrieval | `clients/mcp_search_client.py` → `edumind_mcp_search` server; `clients/tavily_client.py` for YouTube | none (in-process ChromaDB/Tavily RAG removed) |

The five legacy modules (`app/main.py`, `agents/orchestrator.py`,
`agents/tutor.py`, `agents/evaluator.py`, `agents/adaptation_engine.py`) each
carry a `LEGACY` docstring header. Tests for the legacy flow live in
`tests/legacy/` and are not part of the default test run.

## Runtime Entry Points

- `app/api.py` creates the FastAPI app, configures CORS, Prometheus middleware,
  database lifecycle, routers, and legacy `/session/*` endpoints.
- `app/course_api.py` exposes the frontend course/module API under `/api`.
- `app/auth.py` handles Google OAuth, signed session cookies, dev login, and the
  `require_current_user` dependency.
- `app/main.py` is the legacy CLI session entry point and is not the web server.

Docker starts the web service with:

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## Request Flow: Course Creation

1. Frontend calls `POST /api/courses` or the streaming create endpoints.
2. `app/course_api.py` normalizes the setup payload into a service payload.
3. `core/course_service.py` builds the course through roadmap and module planning.
4. `agents/curriculum_architect.py` creates the master roadmap and module plan.
5. `core/curriculum_quality.py` validates scope, dependency order, module
   boundaries, and generated JSON quality.
6. `db/postgres.py` stores the course, modules, roadmap JSON, and decision logs.
7. Streaming endpoints yield progress events as each stage completes.

## Request Flow: Lesson Generation

1. Frontend calls `POST /api/courses/{course_id}/modules/{module_id}/generate`
   or the SSE generation endpoint.
2. `core/course_service.py` loads the owned course and module, checks for
   existing generated content, and builds the lesson prompt context.
3. Lesson generation depends on the configured LLMs. Web-search retrieval, when a
   course enables it, runs through `clients/mcp_search_client.py` against the
   standalone `edumind_mcp_search` server. The old in-process ChromaDB/BGE
   embedding+reranker retrieval path has been removed (see "Live vs legacy code
   paths" below).
4. Generated markdown, optional questions, and optional video metadata are saved
   through `db/postgres.py`.
5. Later reads return the persisted lesson instead of regenerating it.

## Request Flow: Evaluation and Adaptation

There are two evaluation surfaces:

- Legacy interactive agent flow in `agents/evaluator.py`, used by older
  `/session/*` sessions.
- Frontend module evaluation flow in `agents/evaluation_agent.py`, used by
  `/api/courses/{course_id}/modules/{module_id}/evaluation/*`.

Frontend evaluation stages:

1. `start_session()` loads the course/module and creates scoped base questions
   from the saved lesson.
2. `submit_answer()` diagnoses each answer, optionally adds targeted probes, and
   persists the active evaluation session.
3. `_finalize()` computes the final report, writes mastery/skill evidence, saves
   adaptation notes, and returns frontend feedback.

The adaptation engine in `agents/adaptation_engine.py` is used by the legacy
orchestrated session flow. It reasons over evaluation reports and metacognition
signals before recommending whether to move forward, reteach, detour, escalate,
compress, or hold.

## Data Storage

`db/postgres.py` owns schema creation and repository operations. The most
important tables are:

- `students`: learner identity and current preferences.
- `users`: auth user linked to a `student_id`.
- `courses`: frontend course records.
- `course_modules`: module metadata, status, generated lesson markdown.
- `course_roadmaps` and `master_roadmaps`: roadmap JSON.
- `evaluation_sessions`: frontend module evaluation state and reports.
- `evaluation_history`: legacy per-concept evaluation reports.
- `concept_mastery` and `student_skills`: mastery and skill evidence.
- `metacognition`: long-term adaptation profile JSON.
- `decision_log`: agent/course planning audit records.
- `doubt_log` and `module_chat_messages`: learner doubts and side-chat history.

Schema DDL runs during application startup through `init_db()`. Production data
is stored in Docker named volumes and must not be deleted during deployment.

## LLM and External Providers

- `clients/groq_client.py` wraps Groq chat completion, streaming, retry, timeout,
  malformed tool-call recovery, and metrics.
- `clients/tavily_client.py` is used live by `core/course_service.py` to find
  YouTube videos for a module; its results are cached on disk.
- `clients/mcp_search_client.py` is the client for web-search RAG, delegating to
  the standalone `edumind_mcp_search` server.
- The in-process retrieval stack (`db/chromadb_client.py`, `core/rag_pipeline.py`
  and the HyDE/ChromaDB/Tavily/reranker evaluation metrics) has been **removed** —
  it was disabled in production and never served the live flow.

Provider calls must never log API keys, raw prompts, full generated lessons, or
private learner text.

## Observability

- `loguru` is the project logger used across the backend.
- `core/metrics.py` defines Prometheus counters/histograms.
- `core/metrics_middleware.py` records request metrics.
- `/metrics` exposes Prometheus format output.
- `evaluation/` can write session and aggregate evaluation reports.

See [Logging](LOGGING.md) for safe logging rules.
