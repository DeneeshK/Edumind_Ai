from __future__ import annotations

import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from google.auth import transport
from google.oauth2 import id_token as google_id_token
from jose import JWTError, jwt
from starlette.concurrency import run_in_threadpool

from config import settings
from db.postgres import upsert_google_user


router = APIRouter(tags=["auth"])

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
JWT_ALGORITHM = "HS256"
STATE_COOKIE_MAX_AGE_SECONDS = 600


class _HttpxGoogleAuthResponse(transport.Response):
    def __init__(self, response: httpx.Response):
        self._response = response

    @property
    def status(self) -> int:
        return self._response.status_code

    @property
    def headers(self) -> dict[str, str]:
        return dict(self._response.headers)

    @property
    def data(self) -> bytes:
        return self._response.content


class _HttpxGoogleAuthRequest(transport.Request):
    def __call__(
        self,
        url: str,
        method: str = "GET",
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        timeout: int | float | None = None,
        **_: Any,
    ) -> transport.Response:
        with httpx.Client(timeout=timeout or 10.0) as client:
            response = client.request(method, url, content=body, headers=headers)
        return _HttpxGoogleAuthResponse(response)


def _state_cookie_name() -> str:
    return f"{settings.session_cookie_name}_oauth_state"


def _require_auth_config() -> None:
    missing = [
        name
        for name, value in {
            "GOOGLE_CLIENT_ID": settings.google_client_id,
            "GOOGLE_CLIENT_SECRET": settings.google_client_secret,
            "GOOGLE_REDIRECT_URI": settings.google_redirect_uri,
            "FRONTEND_URL": settings.frontend_url,
            "SESSION_SECRET_KEY": settings.session_secret_key,
        }.items()
        if not value
    ]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Missing auth configuration: {', '.join(missing)}",
        )


def _encode_jwt(payload: dict[str, Any], max_age_seconds: int) -> str:
    now = int(time.time())
    return jwt.encode(
        {
            **payload,
            "iat": now,
            "exp": now + max_age_seconds,
        },
        settings.session_secret_key,
        algorithm=JWT_ALGORITHM,
    )


def _decode_jwt(token: str) -> dict[str, Any] | None:
    try:
        payload = jwt.decode(
            token,
            settings.session_secret_key,
            algorithms=[JWT_ALGORITHM],
        )
    except JWTError:
        return None
    return payload if isinstance(payload, dict) else None


def _create_state_token(state: str) -> str:
    return _encode_jwt(
        {
            "typ": "oauth_state",
            "state": state,
        },
        STATE_COOKIE_MAX_AGE_SECONDS,
    )


def _validate_state(request: Request, state: str | None) -> None:
    if not state:
        raise HTTPException(status_code=400, detail="Missing OAuth state")

    state_token = request.cookies.get(_state_cookie_name())
    if not state_token:
        raise HTTPException(status_code=400, detail="Missing OAuth state cookie")

    payload = _decode_jwt(state_token)
    if not payload or payload.get("typ") != "oauth_state":
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    if not secrets.compare_digest(str(payload.get("state") or ""), state):
        raise HTTPException(status_code=400, detail="OAuth state mismatch")


def _create_session_token(user: dict[str, Any]) -> str:
    return _encode_jwt(
        {
            "typ": "session",
            "user_id": user.get("id"),
            "student_id": user.get("student_id"),
            "email": user.get("email"),
            "name": user.get("name"),
            "avatar_url": user.get("avatar_url"),
        },
        settings.session_max_age_seconds,
    )


def _session_user_from_request(request: Request) -> dict[str, Any] | None:
    if not settings.session_secret_key:
        return None

    token = request.cookies.get(settings.session_cookie_name)
    if not token:
        return None

    payload = _decode_jwt(token)
    if not payload or payload.get("typ") != "session":
        return None

    user_id = payload.get("user_id")
    return {
        "id": user_id,
        "user_id": user_id,
        "student_id": payload.get("student_id"),
        "email": payload.get("email"),
        "name": payload.get("name"),
        "avatar_url": payload.get("avatar_url"),
    }


def get_current_user(request: Request) -> dict[str, Any] | None:
    return _session_user_from_request(request)


def require_current_user(request: Request) -> dict[str, Any]:
    user = get_current_user(request)
    if not user or not user.get("student_id"):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


def _verify_google_id_token(token: str) -> dict[str, Any]:
    id_info = google_id_token.verify_oauth2_token(
        token,
        _HttpxGoogleAuthRequest(),
        settings.google_client_id,
    )
    if id_info.get("aud") != settings.google_client_id:
        raise ValueError("Google token audience mismatch")
    return id_info


@router.get("/auth/google/login")
async def google_login():
    _require_auth_config()
    state = secrets.token_urlsafe(32)
    params = {
        "response_type": "code",
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "scope": "openid email profile",
        "state": state,
    }
    response = RedirectResponse(f"{GOOGLE_AUTH_URL}?{urlencode(params)}")
    response.set_cookie(
        key=_state_cookie_name(),
        value=_create_state_token(state),
        max_age=STATE_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
    )
    return response


@router.get("/auth/google/callback")
async def google_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
):
    _require_auth_config()
    if error:
        raise HTTPException(status_code=400, detail=f"Google OAuth error: {error}")
    if not code:
        raise HTTPException(status_code=400, detail="Missing Google OAuth code")

    _validate_state(request, state)

    async with httpx.AsyncClient(timeout=15.0) as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

    if token_response.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail="Failed to exchange Google OAuth code",
        )

    token_payload = token_response.json()
    raw_id_token = token_payload.get("id_token")
    if not raw_id_token:
        raise HTTPException(status_code=400, detail="Google did not return an ID token")

    try:
        id_info = await run_in_threadpool(_verify_google_id_token, raw_id_token)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail="Invalid Google ID token") from exc

    google_sub = str(id_info.get("sub") or "")
    email = str(id_info.get("email") or "")
    name = str(id_info.get("name") or email or "EduMind Student")
    avatar_url = str(id_info.get("picture") or "")
    if not google_sub or not email:
        raise HTTPException(status_code=400, detail="Google profile is missing required fields")

    user = await upsert_google_user(
        google_sub=google_sub,
        email=email,
        name=name,
        avatar_url=avatar_url,
    )

    response = RedirectResponse(settings.frontend_url)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=_create_session_token(user),
        max_age=settings.session_max_age_seconds,
        httponly=True,
        secure=False,
        samesite="lax",
        path="/",
    )
    response.delete_cookie(
        key=_state_cookie_name(),
        path="/",
        samesite="lax",
    )
    return response


@router.get("/api/auth/me")
async def auth_me(request: Request):
    user = _session_user_from_request(request)
    if not user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": user}


@router.post("/api/auth/logout")
async def logout():
    response = JSONResponse({"success": True})
    response.delete_cookie(
        key=settings.session_cookie_name,
        path="/",
        samesite="lax",
    )
    response.delete_cookie(
        key=_state_cookie_name(),
        path="/",
        samesite="lax",
    )
    return response
