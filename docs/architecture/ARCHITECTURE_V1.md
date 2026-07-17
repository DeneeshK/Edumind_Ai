# Architecture — V1 (archived)

> **Historical reference only.** This is the architecture doc as it stood at
> commit `7f1c146`, before the retrieval/prompt/observability/guardrails
> rework described in [ARCHITECTURE_V2](ARCHITECTURE_V2.md). It is kept
> verbatim so the two can be diffed. For the current system, read
> [`../ARCHITECTURE.md`](../ARCHITECTURE.md), which points to V2.
>
> At this point in the project: retrieval ran through an in-process
> ChromaDB/BGE-embedding/reranker stack that was disabled but not removed;
> there was no request tracing; prompts were inline strings with no version
> history or regression suite; and LLM JSON output was consumed via `.get()`
> with no schema validation or fencing against student-controlled text.

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
| `clients/` | External provider clients for Groq and Tavily. ChromaDB client is retained for compatibility while retrieval is disabled. |
| `evaluation/` | Offline and runtime metrics, evaluation reports, scheduler, and evaluation API endpoints. |
| `tests/` | Unit and mocked integration tests for service behavior, routes, auth, config, and report writing. |
| `monitoring/` | Prometheus and Grafana configuration for local/production observability. |

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
3. Lesson generation currently depends on the configured LLMs. Tavily search,
   ChromaDB retrieval, embeddings, and reranking are disabled in the generation
   path, with compatibility modules retained so older imports do not break.
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
- `clients/tavily_client.py` remains available for search helper compatibility.
  Current lesson generation is not Tavily-dependent.
- `db/chromadb_client.py` and `core/rag_pipeline.py` remain as compatibility
  modules while retrieval/embeddings/reranking are disabled.

Provider calls must never log API keys, raw prompts, full generated lessons, or
private learner text.

## Observability

- `loguru` is the project logger used across the backend.
- `core/metrics.py` defines Prometheus counters/histograms.
- `core/metrics_middleware.py` records request metrics.
- `/metrics` exposes Prometheus format output.
- `evaluation/` can write session and aggregate evaluation reports.

See [Logging](LOGGING.md) for safe logging rules.
