# Logging

The backend uses `loguru` for application logging. Most modules import:

```python
from loguru import logger
```

Logging should explain lifecycle events and failures without exposing private
learner content, secrets, or full generated artifacts.

## Current Logging Surfaces

| Area | Examples |
| --- | --- |
| Application lifecycle | Startup, shutdown, database pool initialization, scheduler start/stop. |
| Agent execution | Agent trace records, tool-call execution, routing decisions, fallback paths. |
| Course generation | Planning stages, validation failures, repair attempts, persistence. |
| Lesson generation | Module-level lifecycle, fallback conditions, content saved state. |
| Evaluation | Question generation failure, answer diagnosis failure, final report failure. |
| Database | Pool connection retries, schema readiness, shutdown. |
| External providers | Groq retry, timeout, rate limit, malformed tool-call recovery. |
| Metrics | Prometheus counters and histograms exposed at `/metrics`. |

## Safe Metadata To Log

Prefer short metadata that helps trace operations:

- `course_id`
- `module_id`
- `session_id`
- internal `student_id` when already used in backend logs
- model name
- provider name
- item counts
- status strings
- retry attempt number
- duration/latency
- sanitized exception message or exception type

## Never Log

- API keys
- OAuth tokens
- private SSH keys
- passwords
- authorization headers
- full environment variables
- full database URLs with credentials
- session cookie values
- raw prompts
- full generated lessons
- full LLM responses that may contain user/private data
- raw uploaded files or full document contents
- large retrieved context blocks

When debugging prompt or generation issues, log identifiers, lengths, counts, and
failure categories instead of raw content.

## Level Guide

| Level | Use for |
| --- | --- |
| `debug` | Developer-only details, loop iterations, narrow diagnostics. |
| `info` | Normal lifecycle events that matter operationally. |
| `warning` | Recoverable failures, fallback execution, suspicious empty outputs. |
| `error` | Operation failed and was handled or surfaced. |
| `exception` | Inside exception handlers where stack trace is useful and safe. |

## Noisy Log Avoidance

Avoid logs inside tight loops unless they are debug-level and low volume. Avoid
logging every streamed token, every SSE message, or every generated paragraph.
For streaming, log start/end, counts, and failures.

## Production Log Checks

On EC2:

```bash
cd /home/ubuntu/Edumind_Ai
docker compose logs --tail=200 backend
docker compose logs --tail=120 postgres
sudo tail -n 100 /var/log/nginx/error.log
```

For a live tail:

```bash
docker compose logs -f backend
```

## Adding New Logs

Before adding a new log, check:

1. Does this help diagnose a real production or developer issue?
2. Can it be emitted frequently enough to become noisy?
3. Does it include any raw prompt, user answer, generated lesson, token, cookie,
   API key, or credential-bearing URL?
4. Would a count, id, status, or error type communicate the same information?

If the answer is not clearly safe and useful, do not add the log.
