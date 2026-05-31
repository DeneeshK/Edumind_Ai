# Developer Handover

This document gives a new backend developer the fastest path to understand where
important behavior lives and what must stay stable.

## What This Backend Does

EduMind turns a learner's topic, goal, pace, and profile into a personalized
course. It stores a roadmap and modules, generates lessons, evaluates answers,
updates mastery/adaptation signals, and returns progress data for the frontend.

## Files To Read First

1. `app/api.py` - FastAPI app setup, middleware, lifecycle, legacy sessions.
2. `app/course_api.py` - Frontend `/api/*` routes and request/response flow.
3. `core/course_service.py` - Course creation, lesson generation, chat, and
   module evaluation service layer.
4. `agents/curriculum_architect.py` - Roadmap/module planning pipeline.
5. `agents/evaluation_agent.py` - Frontend module evaluation flow.
6. `agents/adaptation_engine.py` - Legacy adaptive routing decision support.
7. `core/curriculum_quality.py` - Validation, parsing, scope, dependency, and
   repair helpers.
8. `db/postgres.py` - Schema and repository functions.
9. `config.py` - Runtime settings.

## Main Flow Ownership

| Flow | Primary files |
| --- | --- |
| FastAPI startup/shutdown | `app/api.py`, `db/postgres.py`, `evaluation/scheduler.py` |
| Auth/session cookies | `app/auth.py`, auth routes in `app/course_api.py` |
| Course setup payload normalization | `app/course_api.py` |
| Course creation | `core/course_service.py`, `agents/curriculum_architect.py`, `core/roadmap_service.py` |
| Curriculum validation | `core/curriculum_quality.py` |
| Module lesson generation | `core/course_service.py`, `clients/groq_client.py` |
| Module chat/doubts | `core/course_service.py`, `db/postgres.py` |
| Evaluation questions and reports | `agents/evaluation_agent.py`, `db/postgres.py` |
| Legacy live sessions | `app/api.py`, `agents/orchestrator.py`, `agents/tutor.py`, `agents/evaluator.py` |
| Metrics/reporting | `core/metrics.py`, `core/metrics_middleware.py`, `evaluation/` |

## Important Current Design Notes

- The frontend-facing API is course/module based under `/api`.
- Legacy `/session/*` endpoints still exist for interactive agent sessions.
- Lesson generation is currently LLM-first. Tavily, ChromaDB retrieval,
  embeddings, and reranking are not the source of lesson content in the current
  path.
- Compatibility modules for retrieval/search remain because older code imports
  them and tests may still exercise helpers.
- PostgreSQL is the source of truth for courses, modules, generated lesson
  content, evaluations, progress, and adaptation state.
- Generated lesson content is persisted; reopening a module should not regenerate
  content unless the service explicitly decides to.
- Production `.env` exists only on EC2 and must not be committed or overwritten.

## Stability Rules

Coordinate frontend and tests before changing:

- route paths
- request/response field names
- auth cookie behavior
- evaluation session response shape
- course/module status values
- database table or column semantics
- LLM tool schemas
- prompt output JSON contracts
- Docker Compose service names or named volumes

## Database Safety

`db/postgres.py` applies schema DDL at startup. Production runs in Docker Compose
with named volumes. Never delete volumes during deployment.

Unsafe in production:

```bash
docker compose down -v
```

Safe deploy/restart shape:

```bash
docker compose up --build -d
docker compose ps
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/ready
```

## Production Deployment

GitHub Actions deploys on pushes to `main` through
`.github/workflows/deploy-backend.yml`. The workflow SSHes into EC2, resets the
server checkout to `origin/main`, verifies `.env`, rebuilds Compose services,
and fails if health checks fail.

Manual EC2 deploy commands are documented in `DOCKER.md`.

## Before Merging Backend Changes

Run:

```bash
git diff --check
venv/bin/pytest -q
```

For documentation-only Python changes, also run the AST docstring-stripping check
from `docs/TESTING.md`.

Review logs and generated docs for:

- no secrets
- no raw prompts
- no full generated lessons
- no credential-bearing URLs
- no accidental behavior changes

## Common Review Focus

- Does the route still enforce ownership with the authenticated `student_id`?
- Does the service write to the correct course/module row?
- Does evaluation stay grounded in saved lesson content?
- Does fallback behavior preserve user progress?
- Are external provider failures handled without leaking private payloads?
- Are Docker volumes preserved?
