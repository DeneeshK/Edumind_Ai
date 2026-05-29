from __future__ import annotations

import pytest
from pydantic import ValidationError

from config import Settings


pytestmark = pytest.mark.unit


def test_settings_loads_with_fake_test_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setenv("ENVIRONMENT", "test")

    settings = Settings(_env_file=None)

    assert settings.groq_api_key == "fake-groq"
    assert settings.tavily_api_key == "fake-tavily"
    assert settings.database_url.startswith("postgresql://")
    assert settings.dev_auth_enabled is True
    assert settings.environment == "test"


def test_required_config_validation_is_explicit(monkeypatch):
    for name in ("GROQ_API_KEY", "TAVILY_API_KEY", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError) as exc:
        Settings(_env_file=None)

    message = str(exc.value)
    assert "groq_api_key" in message
    assert "tavily_api_key" in message
    assert "database_url" in message


def test_test_environment_does_not_require_google_oauth_secrets(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    monkeypatch.delenv("GOOGLE_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_CLIENT_SECRET", raising=False)

    settings = Settings(_env_file=None)

    assert settings.google_client_id == ""
    assert settings.google_client_secret == ""


def test_cors_origins_are_parsed_from_comma_separated_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setenv(
        "CORS_ORIGINS",
        "http://localhost:5173, http://127.0.0.1:5173, http://localhost:3000, https://edumind.ai",
    )

    settings = Settings(_env_file=None)

    assert settings.cors_origin_list == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://edumind.ai",
    ]


def test_production_cors_rejects_wildcard_with_credentials(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq")
    monkeypatch.setenv("TAVILY_API_KEY", "fake-tavily")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ORIGINS", "*")

    settings = Settings(_env_file=None)

    with pytest.raises(ValueError, match="CORS_ORIGINS"):
        _ = settings.cors_origin_list
