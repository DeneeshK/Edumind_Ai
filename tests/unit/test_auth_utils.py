from __future__ import annotations

from http.cookies import SimpleCookie

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import app.auth as auth


pytestmark = pytest.mark.unit


def _request_with_cookies(cookies: dict[str, str] | None = None) -> Request:
    headers = [(b"host", b"testserver")]
    if cookies:
        jar = SimpleCookie()
        for key, value in cookies.items():
            jar[key] = value
        cookie_header = "; ".join(f"{key}={morsel.value}" for key, morsel in jar.items())
        headers.append((b"cookie", cookie_header.encode("latin-1")))
    return Request({"type": "http", "method": "GET", "path": "/", "headers": headers})


def test_session_token_round_trips_to_current_user(monkeypatch):
    monkeypatch.setattr(auth.settings, "session_secret_key", "test-session-secret")
    monkeypatch.setattr(auth.settings, "session_cookie_name", "edumind_test_session")

    token = auth._create_session_token(
        {
            "id": "user-1",
            "student_id": "student-1",
            "email": "student@example.test",
            "name": "Student One",
            "avatar_url": "",
        }
    )

    request = _request_with_cookies({"edumind_test_session": token})
    user = auth.get_current_user(request)

    assert user is not None
    assert user["id"] == "user-1"
    assert user["student_id"] == "student-1"
    assert user["email"] == "student@example.test"


def test_invalid_session_token_is_ignored(monkeypatch):
    monkeypatch.setattr(auth.settings, "session_secret_key", "test-session-secret")
    monkeypatch.setattr(auth.settings, "session_cookie_name", "edumind_test_session")

    request = _request_with_cookies({"edumind_test_session": "not-a-valid-token"})

    assert auth.get_current_user(request) is None


def test_require_current_user_rejects_missing_session(monkeypatch):
    monkeypatch.setattr(auth.settings, "session_secret_key", "test-session-secret")

    with pytest.raises(HTTPException) as exc:
        auth.require_current_user(_request_with_cookies())

    assert exc.value.status_code == 401


def test_google_id_token_verification_is_mockable(monkeypatch):
    calls = {}
    monkeypatch.setattr(auth.settings, "google_client_id", "google-client-test")

    def fake_verify(token, request, audience):
        calls["token"] = token
        calls["audience"] = audience
        return {"aud": audience, "sub": "google-sub-1", "email": "student@example.test"}

    monkeypatch.setattr(auth.google_id_token, "verify_oauth2_token", fake_verify)

    info = auth._verify_google_id_token("fake-id-token")

    assert info["sub"] == "google-sub-1"
    assert calls == {"token": "fake-id-token", "audience": "google-client-test"}


def test_google_id_token_audience_mismatch_fails(monkeypatch):
    monkeypatch.setattr(auth.settings, "google_client_id", "expected-client")
    monkeypatch.setattr(
        auth.google_id_token,
        "verify_oauth2_token",
        lambda *args, **kwargs: {"aud": "wrong-client"},
    )

    with pytest.raises(ValueError, match="audience mismatch"):
        auth._verify_google_id_token("fake-id-token")
