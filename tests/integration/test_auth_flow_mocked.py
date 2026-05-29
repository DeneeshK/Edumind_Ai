from __future__ import annotations

import pytest

import app.auth as auth


pytestmark = pytest.mark.integration


def test_auth_me_reports_unauthenticated_without_session(api_client):
    response = api_client.get("/api/auth/me")

    assert response.status_code == 200
    assert response.json() == {"authenticated": False, "user": None}


def test_auth_me_reads_mocked_session_cookie(api_client, monkeypatch):
    monkeypatch.setattr(auth.settings, "session_secret_key", "test-session-secret")
    monkeypatch.setattr(auth.settings, "session_cookie_name", "edumind_test_session")
    token = auth._create_session_token(
        {
            "id": "user-test-1",
            "student_id": "student-test-1",
            "email": "student@example.test",
            "name": "Test Student",
            "avatar_url": "",
        }
    )

    api_client.cookies.set("edumind_test_session", token)
    response = api_client.get("/api/auth/me")

    assert response.status_code == 200
    body = response.json()
    assert body["authenticated"] is True
    assert body["user"]["student_id"] == "student-test-1"
