from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    groq_api_key: str = Field(..., description="Groq API key")
    tavily_api_key: str = Field(..., description="Tavily search API key")
    edumind_api_key: str = Field(
        default="",
        description="API key for EduMind API endpoints. Leave empty to disable auth (dev mode)."
    )
    cors_origins: str = "http://localhost:5173"
    dev_auth_enabled: bool = True

    # Database
    database_url: str = Field(
        ..., description="PostgreSQL connection URL (asyncpg format)"
    )
    chromadb_path: str = Field(
        default="./chromadb_data", description="Path to ChromaDB persistent storage"
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
    eval_embed_model: str = "all-MiniLM-L6-v2"
    eval_faithfulness_claim_limit: int = 15
    eval_precision_k: int = 10
    eval_schedule_weekly: bool = True
    eval_schedule_monthly: bool = True
    eval_schedule_timezone: str = "Asia/Kolkata"


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
