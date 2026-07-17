# EduMind AI

**Adaptive learning backend** — turns a learner's goal into a personalized course,
teaches it module by module, evaluates understanding through Socratic questioning
rather than multiple choice, and adapts the next lesson to what the evaluation
actually found.

Live at **[edumindai.org](https://edumindai.org)** · API at
`course-api.edumindai.org` · backend-only repository (frontend deployed
separately, see [Edumind_Ai_frontend](../Edumind_Ai_frontend)).

---

## What it does

1. **Intent capture** — a learner states a topic, goal, prior knowledge, and pace.
   Google OAuth (or dev login) identifies them; their profile is sanitized and
   carried through every downstream prompt.
2. **Curriculum planning** — `CurriculumArchitectAgent` runs a two-call pipeline
   (coverage planner → sequencer) plus an LLM auditor pass, then validates the
   result structurally (prerequisite ordering, scope, dedup) before it's saved.
3. **Lesson generation** — each module's lesson is generated with the learner's
   adaptation history folded in: prior weak concepts, doubt patterns, and stated
   style preference all become concrete instructions in the lesson prompt, not
   just metadata.
4. **Grounded evaluation** — the evaluator diagnoses each free-text answer
   against the actual lesson content (never inventing outside material), decides
   whether the answer is confident, vague, or wrong, and chains a targeted
   follow-up probe off the specific weakness it just found — closer to an
   interviewer digging into an answer than a quiz grader.
5. **Adaptation** — the diagnosis feeds an adaptation summary that changes how
   the *next* module is written: more worked examples, a prerequisite recap, a
   slower pace — whichever the evidence calls for.

## Engineering highlights

This project is built to be read, not just run — the things below are the parts
worth opening first:

- **Model routing, not one-size-fits-all** — reasoning-heavy steps (curriculum
  sequencing, answer diagnosis) run on `openai/gpt-oss-120b`; high-throughput
  generation (lessons, coverage lists) runs on Llama 4 Scout; small/cheap tasks
  use `llama-3.1-8b-instant`. Token counts and per-model cost are tracked in
  Prometheus (`edumind_llm_tokens_total`, `edumind_llm_cost_usd_total`), not
  just assumed.
- **A versioned prompt registry + golden regression suite** — the prompts
  driving curriculum, lesson, and evaluation generation are tracked artifacts
  (`prompts/`) with render-identical snapshot tests, not scattered f-strings. A
  golden eval suite (`evaluation/golden/`) runs real curriculum/diagnosis/lesson
  cases through the production code paths and scores them with the same
  LLM-judge metrics used at runtime — including a standing prompt-injection
  regression case — so a prompt edit that degrades quality can be caught before
  it ships.
- **Guardrails against the actual failure modes of LLM systems** — student
  input is fenced as data (never instructions) before it reaches a grading
  prompt (`core/guardrails.py`), and every LLM JSON response the live flow
  depends on is Pydantic-validated (`core/llm_schemas.py`) with an *observable*
  fallback — malformed output degrades safely and increments a metric, instead
  of silently becoming a wrong default three layers downstream.
- **Retrieval via a decoupled MCP server** — web-search RAG (HyDE + multi-query
  expansion, pgvector, idempotent per-namespace ingestion) runs in a standalone
  `edumind_mcp_search` server, kept out of this API's process so embedding
  models never load into the same memory footprint serving requests. The LLM
  itself decides when a concept is unfamiliar enough to warrant a web search —
  the tool is offered, not forced.
- **Request-scoped tracing, opt-in** — OpenTelemetry spans (GenAI semantic
  conventions) follow one request through every agent, LLM call, and MCP tool
  call, with latency and token cost per step. Off by default
  (`OTEL_ENABLED=false`, zero-cost no-op spans); point it at
  [Phoenix](https://github.com/Arize-ai/phoenix) locally to see the full trace
  tree for a real course-creation or evaluation session.
- **An evaluation framework that scores the system, not just the student** —
  `evaluation/` runs deterministic checks (prerequisite ordering, structural
  validity) alongside LLM-judge metrics (curriculum coverage, lesson quality,
  question quality) at session hooks and writes weekly/monthly aggregate
  reports — separate from the per-student evaluation the product runs live.

## Tech stack

| Layer | Choice |
| --- | --- |
| API | FastAPI, Uvicorn, SSE streaming |
| LLM | Groq (`gpt-oss-120b`, Llama 4 Scout, `llama-3.1-8b-instant`) |
| Retrieval | Standalone MCP server · pgvector · HyDE + multi-query |
| Database | PostgreSQL (asyncpg) |
| Auth | Google OAuth2 + signed session cookies (`python-jose`) |
| Observability | Prometheus + Grafana, OpenTelemetry + Phoenix (opt-in) |
| Validation | Pydantic v2 |
| Frontend | React, Vite, Tailwind, Recharts *(separate repo)* |
| Deployment | Docker Compose on EC2, Nginx (TLS termination) |

## Live vs legacy code paths

- The deployed frontend uses **only** the course-centric `/api/courses` flow:
  `app/course_api.py` → `core/course_service.py` →
  `agents/curriculum_architect.py` + `agents/evaluation_agent.py`, with
  retrieval via `clients/mcp_search_client.py`.
- A second, earlier interactive `/session/*` flow (`app/main.py`,
  `agents/orchestrator.py`, `agents/tutor.py`, `agents/evaluator.py`,
  `agents/adaptation_engine.py`) is **legacy** — a full LLM-orchestrator
  implementation of the same idea, kept as a reference for the queue-based
  interactive-session pattern, tagged `legacy-session` in `/docs`, and marked
  with `LEGACY` docstring headers. It is not used by production traffic.
- The original in-process ChromaDB/BGE-embedding/reranker retrieval stack has
  been removed; it was disabled in production before this cleanup and never
  served the live flow.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full request-flow
breakdown of both paths.

## Quick start

```bash
cp .env.example .env
# fill in GROQ_API_KEY, TAVILY_API_KEY, DATABASE_URL, SESSION_SECRET_KEY
docker compose up --build
```

Or without Docker:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

| | |
| --- | --- |
| API docs | `http://localhost:8000/docs` |
| Health | `http://localhost:8000/health` |
| Readiness | `http://localhost:8000/ready` |
| Metrics | `http://localhost:8000/metrics` |

Dev login (skip Google OAuth locally, `DEV_AUTH_ENABLED=true`):

```bash
curl -X POST http://localhost:8000/api/auth/dev-login \
  -H "Content-Type: application/json" \
  -d '{"email":"student@example.com","name":"Student"}'
```

## Tests

```bash
pytest -q                    # full suite — no real network calls required
pytest tests/unit -q
pytest tests/integration -q  # mocked LLM/auth integration flows
```

Golden prompt-regression suite (needs a real `GROQ_API_KEY`, hits Groq for
real):

```bash
python -m evaluation.golden.run_golden --suite all
```

Runs representative curriculum, diagnosis, and lesson cases through the live
production code paths and the same LLM-judge metrics used at runtime —
including a standing case proving a prompt-injected student answer cannot buy
a passing grade. Currently wired for manual dispatch in CI
(`.github/workflows/golden-evals.yml`) while the case thresholds are being
re-baselined against live Groq rate limits; see the workflow file for status.

## Tracing and cost

Tracing is opt-in and off by default:

```bash
docker compose -f monitoring/docker-compose.monitoring.yml up -d phoenix
OTEL_ENABLED=true OTEL_EXPORTER_ENDPOINT=http://localhost:6006/v1/traces \
  uvicorn app.api:app --port 8000
# hit an endpoint, then open http://localhost:6006
```

<!-- TODO: embed a Phoenix trace screenshot here (HTTP → workflow → agent → LLM/tool spans). -->

Token and cost counters (`edumind_llm_tokens_total`,
`edumind_llm_cost_usd_total`) are always recorded regardless of tracing state,
scraped via `/metrics`. Fill in real Groq per-model prices in
`GROQ_MODEL_PRICES` (`config.py`) before trusting the cost numbers — it ships
with placeholder zeros and skips cost recording for unpriced models.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — versioned index: [V2 (current)](docs/architecture/ARCHITECTURE_V2.md) has full request flows, data model, guardrails, observability; [V1 (archived)](docs/architecture/ARCHITECTURE_V1.md) is the pre-rework snapshot
- [API Reference](docs/API_REFERENCE.md)
- [Environment Variables](docs/ENVIRONMENT.md)
- [Setup](docs/SETUP.md)
- [Testing](docs/TESTING.md)
- [Logging](docs/LOGGING.md)
- [Developer Handover](docs/DEVELOPER_HANDOVER.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Docker & EC2 Deployment](DOCKER.md)
- [Prompt registry](prompts/README.md)

## Production notes

- Backend: `course-api.edumindai.org` · Frontend: `edumindai.org` · EC2 path:
  `/home/ubuntu/Edumind_Ai` · services: `edumind-backend`, `edumind-postgres` ·
  Nginx terminates TLS in front of `127.0.0.1:8000`.
- Never commit `.env`, API keys, OAuth secrets, or a `DATABASE_URL` with real
  credentials.
- Never run `docker compose down -v` in production — it deletes the named
  Postgres volume.
- Never log full prompts, generated lessons, tokens, authorization headers, or
  private learner text — span attributes and traces follow the same rule (ids
  and short excerpts only, see
  [Architecture V2 → Observability](docs/architecture/ARCHITECTURE_V2.md#observability)).
