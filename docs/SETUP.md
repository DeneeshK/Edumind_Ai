# Setup

This guide covers local backend setup. Production deployment details live in
`DOCKER.md`.

## Prerequisites

- Python 3.11
- Docker Engine and Docker Compose v2
- PostgreSQL when running without Docker Compose
- Groq API key for LLM endpoints
- Tavily API key if using search helpers

## Local Environment File

Create a local `.env`:

```bash
cp .env.example .env
```

Edit `.env` and replace placeholder values. At minimum, local development needs:

```text
GROQ_API_KEY=...
TAVILY_API_KEY=...
DATABASE_URL=postgresql://edumind:edumind_password_change_me@localhost:5432/edumind
SESSION_SECRET_KEY=...
```

Use `DEV_AUTH_ENABLED=true` locally if you want to sign in with
`POST /api/auth/dev-login` instead of Google OAuth.

## Run With Docker Compose

Docker Compose starts the backend and PostgreSQL:

```bash
docker compose up --build
```

The backend is available at:

- `http://localhost:8000`
- `http://localhost:8000/docs`
- `http://localhost:8000/health`
- `http://localhost:8000/ready`

Stop containers without deleting volumes:

```bash
docker compose down
```

Do not use `docker compose down -v` unless you intentionally want to delete
local Docker volumes.

## Run Without Docker

Create or reuse a virtual environment, install dependencies, and run uvicorn:

```bash
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

The app will run `init_db()` on startup and create/update tables declared in
`db/postgres.py`.

## Local Auth Options

Development login:

```bash
curl -X POST http://localhost:8000/api/auth/dev-login \
  -H "Content-Type: application/json" \
  -d '{"email":"student@example.com","name":"Student"}'
```

Google OAuth requires valid values for:

- `GOOGLE_CLIENT_ID`
- `GOOGLE_CLIENT_SECRET`
- `GOOGLE_REDIRECT_URI`
- `FRONTEND_URL`
- `SESSION_SECRET_KEY`

The redirect URI in Google Cloud must match `GOOGLE_REDIRECT_URI` exactly.

## Database Notes

`db/postgres.py` owns schema creation. It creates tables for students, users,
courses, modules, roadmaps, evaluations, metacognition, decisions, doubts, and
reports. The schema is applied at app startup.

Local Docker Compose stores Postgres data in the `postgres_data` named volume.
Production also uses Docker named volumes; never delete those during deploy.

## Monitoring Stack

The backend exposes Prometheus metrics at `/metrics`. Local monitoring config is
in `monitoring/`.

To run the optional monitoring stack, inspect:

```bash
monitoring/docker-compose.monitoring.yml
monitoring/prometheus.yml
```

Grafana dashboard provisioning files are under `monitoring/grafana/`.

## Common Developer Commands

```bash
venv/bin/pytest -q
venv/bin/pytest tests/unit -q
venv/bin/pytest tests/integration -q
docker compose logs -f backend
docker compose logs --tail=120 postgres
```

For production deploy and rollback commands, use `DOCKER.md`.
