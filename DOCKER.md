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

Do not use `docker compose down -v` in production. It deletes named volumes,
including the Postgres data volume.

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

## Production Deployment On EC2

Production runs from `/home/ubuntu/Edumind_Ai` on the EC2 host. The production
`.env` file lives only on EC2, is ignored by Git, and must not be committed or
overwritten by deployment.

Manual production deploy command on EC2:

```bash
cd /home/ubuntu/Edumind_Ai
git fetch origin main
git reset --hard origin/main
test -f .env
docker compose up --build -d
docker compose ps
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/ready
curl -f https://course-api.edumindai.org/health
curl -f https://course-api.edumindai.org/ready
```

Never run `docker compose down -v` on production. Keep the Docker named volumes
intact so Postgres data stays safe.

### Continuous Deployment Flow

GitHub Actions deploys on every push to `main` using
`.github/workflows/deploy-backend.yml`. The workflow SSHes into EC2, changes to
`/home/ubuntu/Edumind_Ai`, fetches `origin/main`, resets the server checkout to
`origin/main`, verifies `.env` exists, runs `docker compose up --build -d`, shows
`docker compose ps`, and fails the deployment if any local or public health check
fails.

### GitHub Secrets

Add these in GitHub:

`GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret`

Required repository secrets:

- `EC2_HOST`: EC2 public IP address or DNS name.
- `EC2_USER`: SSH user, usually `ubuntu`.
- `EC2_SSH_KEY`: private key content from your local `.pem` file.

Optional repository secret:

- `EC2_PORT`: SSH port. Omit it to use port `22`.

`EC2_SSH_KEY` must be the full multiline private key content, not the public key
and not a path to the file. Paste it exactly like this shape:

```text
-----BEGIN OPENSSH PRIVATE KEY-----
...
-----END OPENSSH PRIVATE KEY-----
```

Some older `.pem` files use this valid shape instead:

```text
-----BEGIN RSA PRIVATE KEY-----
...
-----END RSA PRIVATE KEY-----
```

The matching public key must already be authorized for `EC2_USER` on the EC2
server, normally in `/home/ubuntu/.ssh/authorized_keys`.

### Production Environment

Keep production values in `/home/ubuntu/Edumind_Ai/.env` only. Required values
include the real API keys, OAuth credentials, session secret, Postgres settings,
and these production URL settings:

```text
FRONTEND_URL=https://edumindai.org
CORS_ORIGINS=https://edumindai.org,https://www.edumindai.org,https://edumind-ai-frontend.vercel.app,http://localhost:5173,http://127.0.0.1:5173
GOOGLE_REDIRECT_URI=https://course-api.edumindai.org/auth/google/callback
RERANKER_ENABLED=false
```

For production, also keep `ENVIRONMENT=production` and do not use `*` in
`CORS_ORIGINS`. If the Postgres volume already exists, keep the existing
`POSTGRES_USER`, `POSTGRES_PASSWORD`, and `POSTGRES_DB` values aligned with that
database.

### Safe Recovery

Check recent logs on EC2:

```bash
cd /home/ubuntu/Edumind_Ai
docker compose logs --tail=200 backend
docker compose logs --tail=120 postgres
sudo tail -n 100 /var/log/nginx/error.log
```

To roll back to a known good commit, SSH into EC2 and run:

```bash
cd /home/ubuntu/Edumind_Ai
git log --oneline -10
git reset --hard <known-good-commit-sha>
docker compose up --build -d
docker compose ps
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/ready
```
