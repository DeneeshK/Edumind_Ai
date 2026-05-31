# Environment Variables

Runtime configuration is loaded by `config.Settings` from environment variables
and the local `.env` file. Docker Compose also reads `.env` for container and
PostgreSQL settings.

Never commit `.env` or real secrets. `.env.example` contains placeholders only.

## Required Application Variables

| Variable | Used by | Purpose | Secret |
| --- | --- | --- | --- |
| `GROQ_API_KEY` | `config.py`, `clients/groq_client.py` | Groq API key for LLM generation, streaming, tool calls, and evaluation. | Yes |
| `TAVILY_API_KEY` | `config.py`, `clients/tavily_client.py` | Tavily search API key. The current lesson path is LLM-first, but the client remains available. | Yes |
| `DATABASE_URL` | `config.py`, `db/postgres.py` | PostgreSQL connection URL. Use `postgresql://...` or `postgresql+asyncpg://...`; the code normalizes asyncpg URLs. | Yes when it contains credentials |

## Runtime and CORS

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | Runtime environment. Set `production` in production. Wildcard CORS origins are rejected in production. |
| `CORS_ORIGINS` | local Vite/React origins | Comma-separated allowed browser origins. Include scheme and host. Do not use `*` in production. |
| `DEV_AUTH_ENABLED` | `true` | Enables `/api/auth/dev-login` for local development. Disable in production. |
| `EDUMIND_API_KEY` | empty | Optional key for legacy protected endpoints using `X-API-Key`. Empty disables that legacy check for development. |

Production-safe URL values currently used by the EC2 deployment:

```text
FRONTEND_URL=https://edumindai.org
CORS_ORIGINS=https://edumindai.org,https://www.edumindai.org,https://edumind-ai-frontend.vercel.app,http://localhost:5173,http://127.0.0.1:5173
GOOGLE_REDIRECT_URI=https://course-api.edumindai.org/auth/google/callback
RERANKER_ENABLED=false
```

These values are not secrets, but they still belong in the production `.env` so
deployments can be changed without editing code.

## Google OAuth and Session Cookies

| Variable | Default | Purpose | Secret |
| --- | --- | --- | --- |
| `GOOGLE_CLIENT_ID` | empty | OAuth client id used for Google login and ID-token audience checks. | No |
| `GOOGLE_CLIENT_SECRET` | empty | OAuth client secret used during authorization-code exchange. | Yes |
| `GOOGLE_REDIRECT_URI` | `http://localhost:8000/auth/google/callback` | Must match the OAuth redirect URI configured in Google Cloud. | No |
| `FRONTEND_URL` | `http://localhost:5173` | Browser redirect target after successful Google login. | No |
| `SESSION_SECRET_KEY` | empty | Secret used to sign backend session and OAuth state cookies. Use a strong unique value in production. | Yes |
| `SESSION_COOKIE_NAME` | `edumind_session` | Name of the signed session cookie. | No |
| `SESSION_MAX_AGE_SECONDS` | `604800` | Session cookie lifetime in seconds. | No |

Production cookies are set with `secure=True`, `httponly=True`, and
`samesite="none"` in OAuth routes so cross-site frontend/backend domains work
over HTTPS.

## Database and Storage

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | required | Application PostgreSQL URL. Docker Compose overrides this inside the backend container to use the `postgres` service hostname. |
| `DB_POOL_SIZE` | `20` | Maximum asyncpg pool size. |
| `CHROMADB_PATH` | `./chromadb_data` | Local ChromaDB storage path retained for compatibility. Current generation does not depend on retrieval. |

Docker Compose-only PostgreSQL variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `POSTGRES_USER` | `edumind` | PostgreSQL user created by the container. |
| `POSTGRES_PASSWORD` | `edumind_password` | PostgreSQL password. Treat as secret. |
| `POSTGRES_DB` | `edumind` | PostgreSQL database name. |
| `POSTGRES_PORT` | `5432` | Host port mapped to container port 5432. |
| `BACKEND_PORT` | `8000` | Host port mapped to the FastAPI container. |

Do not change production Postgres credentials casually after a volume exists.
The named volume contains the initialized database.

## Model Selection

| Variable | Default | Purpose |
| --- | --- | --- |
| `REASONING_MODEL` | `openai/gpt-oss-120b` | Planning, sequencing, routing, and deeper reasoning tasks. |
| `GENERATION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | General content generation and dense extraction. |
| `ADAPTATION_MODEL` | `openai/gpt-oss-120b` | Adaptation and evaluation reasoning where configured. |
| `LESSON_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Lesson content generation. |
| `SMALL_TASK_MODEL` | `llama-3.1-8b-instant` | Lightweight extraction and utility tasks. |
| `GROQ_TIMEOUT_SECONDS` | `120` | Per-request LLM timeout. |
| `GROQ_MAX_RETRIES` | `3` | Retry count for retryable Groq failures. |

Model names are not secrets. API keys are.

## Learning Defaults

| Variable | Default | Purpose |
| --- | --- | --- |
| `MASTERY_THRESHOLD_FAST` | `0.60` | Minimum mastery score for fast pace advancement. |
| `MASTERY_THRESHOLD_MEDIUM` | `0.72` | Minimum mastery score for medium pace advancement. |
| `MASTERY_THRESHOLD_DEEP` | `0.85` | Minimum mastery score for deep pace advancement. |
| `DEFAULT_LESSON_MINUTES` | `10` | Default target lesson length. |
| `DEFAULT_FATIGUE_THRESHOLD_MINUTES` | `25` | Default fatigue threshold used by adaptation state. |

## Evaluation Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `EVAL_ENABLED` | `true` in code, `false` in `.env.example` | Enables runtime evaluation runner/scheduler integration. Keep disabled locally unless intentionally testing it. |
| `EVAL_JUDGE_MODEL` | `llama-3.1-8b-instant` | LLM judge model for evaluation metrics. |
| `EVAL_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model name used by evaluation metrics. |
| `EVAL_FAITHFULNESS_CLAIM_LIMIT` | `15` | Maximum claims checked in faithfulness metrics. |
| `EVAL_PRECISION_K` | `10` | Top-k cutoff for precision-style metrics. |
| `EVAL_SCHEDULE_WEEKLY` | `true` | Enables weekly scheduled aggregate reports when evaluation scheduling is active. |
| `EVAL_SCHEDULE_MONTHLY` | `true` | Enables monthly scheduled aggregate reports when evaluation scheduling is active. |
| `EVAL_SCHEDULE_TIMEZONE` | `Asia/Kolkata` | Timezone used by scheduled evaluation jobs. |

## Compatibility Flags

| Variable | Default | Purpose |
| --- | --- | --- |
| `RERANKER_ENABLED` | `false` | Compatibility flag from the previous retrieval/reranking path. Keep `false` while lesson generation is LLM-first. |

## Production `.env` Handling

On EC2, the production file lives at:

```bash
/home/ubuntu/Edumind_Ai/.env
```

Deployment must verify that file exists but must not overwrite it. GitHub
Actions should not store the contents of `.env`; it only stores SSH connection
secrets needed to reach EC2.
