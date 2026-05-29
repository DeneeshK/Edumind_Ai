# Docker Usage

This Docker setup is backend-only. It runs the FastAPI backend with a local
PostgreSQL service through Docker Compose.

## Prerequisites

- Docker Engine
- Docker Compose v2 (`docker compose`). If your machine only has the legacy
  binary, use `docker-compose` with the same subcommands.
- Real local values for `GROQ_API_KEY` and `TAVILY_API_KEY` if you want to use
  AI/search-powered endpoints

## Environment

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Edit `.env` and replace placeholder API keys/secrets. Do not commit `.env`.

For Docker Compose, the backend database URL is set to the Docker service name:

```text
postgresql://<POSTGRES_USER>:<POSTGRES_PASSWORD>@postgres:5432/<POSTGRES_DB>
```

The `DATABASE_URL` in `.env.example` is mainly for non-Docker local runs.

## Build The Backend Image

```bash
docker build -t edumind-backend .
```

To run that image directly, provide a `.env` file and make sure `DATABASE_URL`
points to a PostgreSQL database reachable from inside the container:

```bash
docker run --env-file .env -p 8000:8000 edumind-backend
```

For normal local development with Postgres included, use Docker Compose instead.

## Run Backend + Postgres

```bash
docker compose up --build
```

The backend will be available at:

- API base: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Dependency readiness check: `http://localhost:8000/ready`
- Metrics: `http://localhost:8000/metrics`

## Logs

```bash
docker compose logs backend
docker compose logs postgres
docker compose logs -f backend
```

## Stop Containers

```bash
docker compose down
```

To remove the Postgres and ChromaDB named volumes too:

```bash
docker compose down -v
```

## Rebuild After Code Or Dependency Changes

```bash
docker compose up --build
```

For a clean rebuild without layer cache:

```bash
docker compose build --no-cache backend
docker compose up
```

## Run Tests Locally

The existing pytest suite is still run from the local Python environment:

```bash
venv/bin/pytest -q
venv/bin/pytest tests/unit -q
venv/bin/pytest tests/integration -q
```
