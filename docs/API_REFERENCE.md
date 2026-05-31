# API Reference

The backend exposes three groups of HTTP endpoints:

- Public operational endpoints on the root FastAPI app.
- Legacy live-session endpoints under `/session/*`.
- Frontend course/module endpoints under `/api/*`.

Interactive OpenAPI documentation is available at `/docs` when the app is
running.

## Authentication Model

Frontend course endpoints use cookie authentication through `require_current_user`
unless noted otherwise.

- Production users authenticate through Google OAuth.
- Development can use `/api/auth/dev-login` when `DEV_AUTH_ENABLED=true`.
- Legacy diagnostic endpoints that include `dependencies=[Depends(verify_api_key)]`
  use the `X-API-Key` header when `EDUMIND_API_KEY` is configured.

Do not pass API keys or session tokens in query strings.

## Operational Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/health` | None | Lightweight process health check. |
| `GET` | `/ready` | None | Dependency readiness check, including database availability. |
| `GET` | `/metrics` | None | Prometheus metrics endpoint, excluded from OpenAPI schema. |

## Google OAuth and Session Endpoints

Defined in `app/auth.py`.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/auth/google/login` | None | Redirects the browser to Google OAuth and stores a signed state cookie. |
| `GET` | `/auth/google/callback` | Google callback | Verifies OAuth state, exchanges the code, creates/updates the user, and sets the backend session cookie. |
| `GET` | `/api/auth/me` | Cookie optional | Returns `{ authenticated, user }` for the current browser session. |
| `POST` | `/api/auth/logout` | Cookie optional | Clears the session and OAuth state cookies. |

## Frontend Auth Endpoints

Defined in `app/course_api.py`.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/auth/dev-login` | Dev-only | Creates or updates a development user when `DEV_AUTH_ENABLED=true`. |
| `GET` | `/api/auth/me` | Cookie optional | Returns current session user for the frontend. |
| `POST` | `/api/auth/logout` | Cookie optional | Clears the frontend session cookie. |

`POST /api/auth/dev-login` accepts:

```json
{
  "email": "student@example.com",
  "name": "Student",
  "avatar_url": ""
}
```

## Course Endpoints

Defined in `app/course_api.py`.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/courses` | Cookie | Lists courses owned by the current student. |
| `POST` | `/api/courses` | Cookie | Creates a course from guided setup fields and returns the saved course. |
| `POST` | `/api/courses/create-intent` | Cookie | Registers a course-creation job for later streaming. |
| `GET` | `/api/courses/{course_id}` | Cookie | Returns one owned course with current progress fields. |
| `GET` | `/api/courses/{course_id}/roadmap` | Cookie | Returns the saved roadmap JSON for the owned course. |
| `POST` | `/api/courses/{course_id}/roadmap/regenerate` | Cookie | Regenerates roadmap data for an owned course. |
| `GET` | `/api/courses/{course_id}/report` | Cookie | Returns the saved completion report or generates one when missing. |

`POST /api/courses` accepts the current guided setup shape:

```json
{
  "student_id": "ignored-for-cookie-auth",
  "topic": "Python",
  "goal": "Build backend APIs",
  "pace": "medium",
  "prior_knowledge": "",
  "name": "Student",
  "profile": {},
  "duration_value": 4,
  "duration_unit": "weeks",
  "hours_per_day": 1,
  "current_level": "basic",
  "goal_description": "Learn practical backend development",
  "deadline": "",
  "preferred_teaching_style": "example_first",
  "assessment_preference": "short quizzes"
}
```

The service normalizes this payload before course generation. `student_id` is
derived from the authenticated session for protected frontend routes.

## Module Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/courses/{course_id}/modules` | Cookie | Lists modules with status, recommendation state, and latest evaluation summary. |
| `GET` | `/api/courses/{course_id}/modules/{module_id}` | Cookie | Returns one owned module and any saved generated lesson. |
| `POST` | `/api/courses/{course_id}/modules/{module_id}/generate` | Cookie | Generates and saves lesson content for a module. |
| `POST` | `/api/courses/{course_id}/modules/{module_id}/complete` | Cookie | Marks the module complete and recalculates course progress. |
| `GET` | `/api/courses/{course_id}/modules/{module_id}/questions` | Cookie | Returns saved/generated module questions. |
| `GET` | `/api/courses/{course_id}/modules/{module_id}/next` | Cookie | Returns the next module by module index. |
| `GET` | `/api/courses/{course_id}/modules/{module_id}/previous` | Cookie | Returns the previous module by module index. |

`POST /generate` writes generated markdown and metadata to `course_modules`.
Repeated reads return persisted content rather than forcing regeneration.

## Module Chat and Doubts

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/courses/{course_id}/modules/{module_id}/chat` | Cookie | Answers a side-chat question and records doubt signals. |
| `GET` | `/api/courses/{course_id}/modules/{module_id}/chat-history` | Cookie | Returns saved chat messages for the current student and module. |

Chat requests accept:

```json
{
  "message": "I do not understand this example."
}
```

The response may include related concepts, possible missing prerequisites, and a
doubt type used later for adaptation.

## Evaluation Endpoints

The modern frontend evaluation flow is session-based.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/api/courses/{course_id}/modules/{module_id}/evaluation/start` | Cookie | Starts an evaluation session and returns base questions. |
| `POST` | `/api/courses/{course_id}/modules/{module_id}/evaluation/{session_id}/answer` | Cookie | Submits one answer, optionally returns a probe/next question, and finalizes when complete. |
| `GET` | `/api/courses/{course_id}/modules/{module_id}/evaluation/latest` | Cookie | Returns the latest evaluation session for the module. |
| `GET` | `/api/courses/{course_id}/modules/{module_id}/evaluation/latest-full` | Cookie | Returns the latest completed evaluation report with questions and answers. |
| `GET` | `/api/courses/{course_id}/modules/{module_id}/evaluation/{session_id}/report` | Cookie | Returns the final evaluation report for a completed session. |
| `POST` | `/api/courses/{course_id}/modules/{module_id}/evaluate` | Cookie | Legacy one-shot answer evaluation endpoint. |

Answer submission accepts:

```json
{
  "question_id": "q1",
  "answer_text": "My answer",
  "confidence": 3
}
```

The evaluation agent saves progress after each answer so reconnecting clients
can recover the session.

## Student Progress Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/students/me/progress` | Cookie | Returns the authenticated student's dashboard-style progress. |
| `GET` | `/api/students/{student_id}/dashboard` | Cookie | Returns dashboard data for the authenticated student's id. |
| `GET` | `/api/students/{student_id}/skills` | Cookie | Returns skill graph/list nodes for the authenticated student's id. |
| `GET` | `/api/students/{student_id}/skills/categorized` | Cookie | Returns categorized skills for the authenticated student's id. |
| `GET` | `/api/students/{student_id}/doubts` | Cookie | Returns doubt history for the authenticated student's id. |
| `GET` | `/api/students/{student_id}/courses` | Cookie | Returns course history for the authenticated student's id. |

Routes with a `student_id` path parameter reject access when the id does not
match the authenticated user's `student_id`.

## Streaming Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/stream/courses/create` | Cookie | Streams course creation from query-string setup fields. |
| `GET` | `/api/stream/courses/create/{job_id}` | Cookie | Streams a previously registered course creation job. |
| `GET` | `/api/stream/courses/{course_id}/create` | Cookie | Replays existing course creation/module state for reconnecting clients. |
| `GET` | `/api/stream/courses/{course_id}/modules/{module_id}/generate` | Cookie | Streams module lesson generation events. |

Streaming responses use `text/event-stream` and emit structured event payloads
through the `_event_stream()` helper.

## Legacy Live Session Endpoints

The legacy session API remains for interactive agent sessions. New frontend
course screens primarily use the `/api/courses/*` endpoints.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `POST` | `/session/start` | Rate limited | Starts or resumes a live adaptive session. |
| `GET` | `/session/stream/{session_id}` | None | Streams session messages and questions over SSE. |
| `POST` | `/session/answer` | None | Submits an answer to the active live session. |
| `POST` | `/session/confidence` | None | Submits a 1-5 confidence score. |
| `POST` | `/session/chat` | None | Submits side-chat text to the live session. |
| `POST` | `/session/end/{session_id}` | None | Ends an in-memory live session. |
| `GET` | `/session/status/{session_id}` | None | Returns current in-memory live session state. |
| `GET` | `/student/{student_id}/progress` | `X-API-Key` when configured | Returns legacy progress data. |
| `GET` | `/session/trace/{session_id}` | `X-API-Key` when configured | Returns trace data for active legacy sessions. |

## Evaluation Metrics API

Defined in `evaluation/api_router.py` with prefix `/eval`.

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/eval/session/{session_id}` | None | Returns a saved evaluation session report. |
| `GET` | `/eval/metrics` | None | Returns metric run rows filtered by optional query parameters. |
| `GET` | `/eval/aggregated` | None | Returns recent weekly or monthly aggregate reports. |
| `POST` | `/eval/run-manual/{session_id}` | None | Triggers manual evaluation metric collection for a session. |

## Debug Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| `GET` | `/api/debug/courses/{course_id}/decision-log` | Cookie | Returns recent agent decision logs for an owned course. |
| `GET` | `/api/debug/session/{session_id}/trace` | Cookie | Returns active session trace metadata. |

Debug responses must not include secrets, OAuth tokens, API keys, or raw private
LLM payloads.
