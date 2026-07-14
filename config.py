"""Application settings loaded from environment variables and the local `.env` file."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_origin_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    """Return a de-duplicated list of CORS origins from string or sequence input."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        text = str(value).strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        raw_items = text.split(",")

    origins: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        origin = str(item).strip().strip("\"'")
        if origin and origin not in seen:
            origins.append(origin)
            seen.add(origin)
    return origins


class Settings(BaseSettings):
    """Typed runtime configuration for API keys, auth, storage, models, and evaluation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Runtime
    environment: str = "development"

    # API Keys
    groq_api_key: str = Field(..., description="Groq API key")
    tavily_api_key: str = Field(..., description="Tavily search API key")
    edumind_api_key: str = Field(
        default="",
        description="API key for EduMind API endpoints. Leave empty to disable auth (dev mode)."
    )
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000"
    dev_auth_enabled: bool = True

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse CORS_ORIGINS and reject wildcard origins in production."""
        origins = _parse_origin_list(self.cors_origins)
        env = (self.environment or "").strip().lower()
        if env in {"prod", "production"} and "*" in origins:
            raise ValueError("CORS_ORIGINS cannot contain '*' when ENVIRONMENT=production")
        return origins

    # Google OAuth + app session cookies
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"
    frontend_url: str = "http://localhost:5173"
    session_secret_key: str = ""
    session_cookie_name: str = "edumind_session"
    session_max_age_seconds: int = 604800

    # Database
    database_url: str = Field(
        ..., description="PostgreSQL connection URL (asyncpg format)"
    )
    # Retained after ChromaDB was removed: this directory is now used only as the
    # on-disk Tavily response cache location (clients/tavily_client.py). The env
    # key stays CHROMADB_PATH for deployment compatibility.
    chromadb_path: str = Field(
        default="./chromadb_data", description="On-disk cache directory (Tavily response cache)"
    )

    # Groq Model Names
    #
    # reasoning_model: used for sequencing (prerequisite ordering) and auditing.
    #   openai/gpt-oss-120b — strongest reasoning on Groq, 8K TPM.
    #   Used wherever deep logical reasoning matters most.
    #
    # generation_model: used for coverage planning and dense extraction.
    #   meta-llama/llama-4-scout-17b-16e-instruct — 30K TPM, broad knowledge,
    #   ideal for "list everything this subject needs" calls.
    #
    # adaptation_model: used by the adaptation engine and tutor agent.
    #   openai/gpt-oss-120b — same quality reasoning for personalization decisions.
    #
    # lesson_model: used for lesson content generation (streaming).
    #   meta-llama/llama-4-scout-17b-16e-instruct — fast, high TPM, fluent prose.
    #
    # small_task_model: used for extraction and lightweight tasks.
    #   llama-3.1-8b-instant — fastest, lowest cost.
    reasoning_model: str = "openai/gpt-oss-120b"
    generation_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    adaptation_model: str = "openai/gpt-oss-120b"
    lesson_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    small_task_model: str = "llama-3.1-8b-instant"

    # Agent Behaviour
    groq_timeout_seconds: int = 120
    groq_max_retries: int = 3

    # Connection Pool
    db_pool_size: int = 20

    # Mastery Thresholds
    mastery_threshold_fast: float = 0.60    # was 0.65
    mastery_threshold_medium: float = 0.72  # was 0.75
    mastery_threshold_deep: float = 0.85    # correct

    # Lesson Defaults
    default_lesson_minutes: int = 10
    default_fatigue_threshold_minutes: int = 25

    # Evaluation settings
    eval_enabled: bool = True
    eval_judge_model: str = "llama-3.1-8b-instant"
    eval_faithfulness_claim_limit: int = 15
    eval_schedule_weekly: bool = True
    eval_schedule_monthly: bool = True
    eval_schedule_timezone: str = "Asia/Kolkata"

    # ── Web-search RAG via the MCP server (client side only) ──────────────────
    # The heavy lifting (Tavily, MiniLM embeddings, pgvector) lives in the
    # standalone edumind_mcp_search server so this API stays RAM-light. Here we
    # only hold what the MCP *client* needs. The per-course web-search toggle
    # gates whether these tools are offered to the agent at all — when a course
    # has it OFF, none of this runs.
    mcp_search_server_url: str = "http://127.0.0.1:8900/sse"
    mcp_search_enabled: bool = True         # master kill-switch for the MCP client
    web_search_default_on: bool = False     # default toggle when the client omits it
    rag_top_k: int = 5                      # chunks to request from the server

    # ── OpenTelemetry tracing (opt-in, off by default) ────────────────────────
    # When otel_enabled is False (the default) no exporter is installed and every
    # tracing code path is a zero-cost no-op — prod is unaffected until opted in.
    # The endpoint points at Phoenix's OTLP/HTTP collector by default; see
    # monitoring/docker-compose.monitoring.yml and docs/ARCHITECTURE.md.
    otel_enabled: bool = False
    otel_exporter_endpoint: str = "http://localhost:6006/v1/traces"
    otel_service_name: str = "edumind-api"


# ── Groq model pricing (USD per 1,000,000 tokens) ─────────────────────────────
#
# ⚠️  MAINTAINER: THESE ARE PLACEHOLDER ZEROS — NOT REAL PRICES. ⚠️
# Fill in the current Groq per-token pricing for each model from
# https://groq.com/pricing before relying on the edumind_llm_cost_usd_total
# metric or the `gen_ai.usage.cost_usd` span attribute. Each entry is
#     model_name: (usd_per_1m_input_tokens, usd_per_1m_output_tokens)
# Cost recording is SKIPPED for any model whose price pair is (0.0, 0.0), so
# leaving these at zero simply disables cost accounting for that model — it
# never invents a number. Do NOT commit guessed prices.
GROQ_MODEL_PRICES: dict[str, tuple[float, float]] = {
    "openai/gpt-oss-120b": (0.0, 0.0),                          # TODO: fill in Groq price
    "meta-llama/llama-4-scout-17b-16e-instruct": (0.0, 0.0),    # TODO: fill in Groq price
    "llama-3.1-8b-instant": (0.0, 0.0),                         # TODO: fill in Groq price
}


def compute_llm_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """Return estimated USD cost for a call, or None if the model is unpriced.

    Returns None when the model is absent from GROQ_MODEL_PRICES or priced at
    (0.0, 0.0) — callers use that to SKIP cost recording rather than emit a
    misleading $0.00. Prices are per 1,000,000 tokens.
    """
    price = GROQ_MODEL_PRICES.get(model)
    if not price:
        return None
    in_price, out_price = price
    if in_price == 0.0 and out_price == 0.0:
        return None
    return (prompt_tokens / 1_000_000) * in_price + (
        completion_tokens / 1_000_000
    ) * out_price


try:
    settings = Settings()
except Exception as _e:
    import sys
    print(
        f"\n❌ EduMind config error: {_e}"
        f"\n   Make sure your .env file exists and contains:"
        f"\n   GROQ_API_KEY=..."
        f"\n   TAVILY_API_KEY=..."
        f"\n   DATABASE_URL=postgresql://user:pass@localhost:5432/edumind\n"
    )
    sys.exit(1)
