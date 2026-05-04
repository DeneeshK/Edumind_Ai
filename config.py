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

    # Database
    database_url: str = Field(
        ..., description="PostgreSQL connection URL (asyncpg format)"
    )
    chromadb_path: str = Field(
        default="./chromadb_data", description="Path to ChromaDB persistent storage"
    )

    # Groq Model Names
    reasoning_model: str = "llama-3.3-70b-versatile"
    generation_model: str = "llama-3.1-8b-instant"

    # Agent Behaviour
    groq_timeout_seconds: int = 30
    groq_max_retries: int = 3

    # Connection Pool
    db_pool_size: int = 5

    # Mastery Thresholds
    mastery_threshold_fast: float = 0.60    # was 0.65
    mastery_threshold_medium: float = 0.72  # was 0.75
    mastery_threshold_deep: float = 0.85    # correct

    # Lesson Defaults
    default_lesson_minutes: int = 10
    default_fatigue_threshold_minutes: int = 25


settings = Settings()
