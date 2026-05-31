# Troubleshooting

Use this guide when the backend fails to start, deploy, authenticate, generate
content, or pass health checks.

## Health Check Fails

Check local process health:

```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/ready
```

If `/health` fails, the FastAPI process is not reachable. If `/ready` fails, the
process may be running but a dependency such as PostgreSQL is unavailable.

Docker checks:

```bash
docker compose ps
docker compose logs --tail=200 backend
docker compose logs --tail=120 postgres
```

## Backend Container Does Not Start

Common causes:

- Missing `.env`.
- Invalid `DATABASE_URL`.
- Missing required `GROQ_API_KEY`, `TAVILY_API_KEY`, or `DATABASE_URL`.
- PostgreSQL container is still starting.
- Dependency import failure after a code change.

Commands:

```bash
docker compose ps
docker compose logs --tail=200 backend
docker compose logs --tail=120 postgres
```

The database initialization has retry/backoff logic, so short Postgres startup
delays should recover automatically.

## Database Connection Errors

For Docker Compose, the backend container must use the Postgres service name:

```text
postgresql://<POSTGRES_USER>:<POSTGRES_PASSWORD>@postgres:5432/<POSTGRES_DB>
```

For non-Docker local runs, use a host reachable from the local Python process,
often:

```text
postgresql://edumind:edumind_password_change_me@localhost:5432/edumind
```

Check the Postgres container:

```bash
docker compose logs --tail=120 postgres
docker compose exec postgres pg_isready -U edumind -d edumind
```

Adjust user/database names if your `.env` uses different values.

## CORS or Cookie Auth Fails

Check:

- `FRONTEND_URL` matches the browser frontend origin.
- `CORS_ORIGINS` contains the exact frontend origin, including scheme and port.
- `ENVIRONMENT=production` does not use `*` in `CORS_ORIGINS`.
- Production requests use HTTPS so secure cookies can be stored.
- Google OAuth redirect URI exactly matches `GOOGLE_REDIRECT_URI`.
- `SESSION_SECRET_KEY` is stable across restarts.

Useful auth endpoint:

```bash
curl -i http://localhost:8000/api/auth/me
```

For local development, use `DEV_AUTH_ENABLED=true` and `/api/auth/dev-login`.

## Google OAuth Callback Fails

Common causes:

- OAuth client id or secret is missing.
- `GOOGLE_REDIRECT_URI` differs from the Google Cloud OAuth setting.
- Browser rejected the state cookie due to domain/HTTPS mismatch.
- `SESSION_SECRET_KEY` changed between login and callback.

Do not log or paste OAuth codes, ID tokens, or session cookies into tickets.
Use status codes and sanitized error messages instead.

## Lesson Generation Fails

Check:

- `GROQ_API_KEY` is set and valid.
- Configured model names are accepted by Groq.
- Provider rate limits have not been exceeded.
- The course/module belongs to the authenticated user.
- The module row exists in `course_modules`.

Useful logs:

```bash
docker compose logs --tail=200 backend
```

The current lesson path is LLM-first. Tavily search, ChromaDB retrieval,
embeddings, and reranking are not required for normal lesson generation.

## Evaluation Fails To Start

Evaluation needs module lesson content. The route attempts to generate missing
lesson content before starting evaluation, then lets the evaluation agent decide
whether enough content exists.

Check:

- The module exists and belongs to the authenticated user.
- Lesson generation is succeeding.
- `evaluation_sessions` writes are succeeding.
- Groq calls are not timing out or rate limited.

## Deployment Fails In GitHub Actions

The deployment workflow should fail on SSH errors, missing `.env`, Compose build
errors, or health-check failures.

Check GitHub repository secrets:

- `EC2_HOST`
- `EC2_USER`
- `EC2_SSH_KEY`
- optional `EC2_PORT`

`EC2_SSH_KEY` must contain the full private key content from the local `.pem`
file, including BEGIN/END lines. It is not a file path.

On EC2:

```bash
cd /home/ubuntu/Edumind_Ai
git status --short
docker compose ps
docker compose logs --tail=200 backend
docker compose logs --tail=120 postgres
sudo tail -n 100 /var/log/nginx/error.log
```

## Public Domain Health Fails But Local Health Passes

If these pass:

```bash
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/ready
```

but these fail:

```bash
curl -f https://course-api.edumindai.org/health
curl -f https://course-api.edumindai.org/ready
```

then inspect Nginx and SSL:

```bash
sudo nginx -t
sudo systemctl status nginx
sudo tail -n 100 /var/log/nginx/error.log
```

Also confirm DNS points to the EC2 instance and Certbot certificates are valid.

## Safe Rollback

On EC2:

```bash
cd /home/ubuntu/Edumind_Ai
git log --oneline -10
git reset --hard <known-good-commit-sha>
docker compose up --build -d
docker compose ps
curl -f http://127.0.0.1:8000/health
curl -f http://127.0.0.1:8000/ready
curl -f https://course-api.edumindai.org/health
curl -f https://course-api.edumindai.org/ready
```

Do not run `docker compose down -v` during rollback.
