from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient


TEST_ENV = {
    "ENVIRONMENT": "test",
    "GROQ_API_KEY": "test-groq-key",
    "TAVILY_API_KEY": "test-tavily-key",
    "GOOGLE_CLIENT_ID": "test-google-client-id",
    "GOOGLE_CLIENT_SECRET": "test-google-client-secret",
    "GOOGLE_REDIRECT_URI": "http://testserver/auth/google/callback",
    "FRONTEND_URL": "http://testserver",
    "SESSION_SECRET_KEY": "test-session-secret",
    "SESSION_COOKIE_NAME": "edumind_test_session",
    "EDUMIND_API_KEY": "",
    "DATABASE_URL": "postgresql://test:test@localhost:5432/edumind_test",
    "CHROMADB_PATH": str(Path(tempfile.gettempdir()) / "edumind-test-chromadb"),
    "EVAL_ENABLED": "false",
}

os.environ.update(TEST_ENV)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _block_external_network(monkeypatch: pytest.MonkeyPatch):
    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any):
        host = ""
        if isinstance(address, tuple) and address:
            host = str(address[0])
        elif isinstance(address, str):
            host = address

        if host in {"127.0.0.1", "::1", "localhost"} or host.startswith("/"):
            return real_connect(self, address)
        raise AssertionError(f"External network access blocked during tests: {address!r}")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def load_fixture(fixture_dir: Path):
    def _load(name: str) -> dict[str, Any]:
        return json.loads((fixture_dir / name).read_text(encoding="utf-8"))

    return _load


@pytest.fixture
def sample_course(load_fixture):
    return load_fixture("sample_course.json")


@pytest.fixture
def sample_roadmap(load_fixture):
    return load_fixture("sample_roadmap.json")


@pytest.fixture
def sample_student_state(load_fixture):
    return load_fixture("sample_student_state.json")


@pytest.fixture
def current_user() -> dict[str, Any]:
    return {
        "id": "user-test-1",
        "user_id": "user-test-1",
        "student_id": "student-test-1",
        "email": "student@example.test",
        "name": "Test Student",
        "avatar_url": "",
    }


@pytest.fixture
def api_app():
    from app.api import app

    app.dependency_overrides.clear()
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def api_client(api_app):
    client = TestClient(api_app)
    try:
        yield client
    finally:
        client.close()
