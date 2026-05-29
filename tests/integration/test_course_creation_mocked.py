from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import app.course_api as course_api
from app.auth import require_current_user


pytestmark = pytest.mark.integration


def test_protected_courses_route_rejects_without_auth(api_client):
    response = api_client.get("/api/courses")

    assert response.status_code == 401


def test_protected_courses_route_accepts_mocked_user(api_app, api_client, current_user, monkeypatch):
    api_app.dependency_overrides[require_current_user] = lambda: current_user
    monkeypatch.setattr(
        course_api,
        "list_courses",
        AsyncMock(return_value=[{"id": "course-python-1", "title": "Python Fundamentals"}]),
    )

    response = api_client.get("/api/courses")

    assert response.status_code == 200
    assert response.json()["courses"][0]["id"] == "course-python-1"


def test_course_creation_endpoint_uses_mocked_course_service(
    api_app,
    api_client,
    current_user,
    sample_course,
    sample_roadmap,
    monkeypatch,
):
    api_app.dependency_overrides[require_current_user] = lambda: current_user
    course = {**sample_course, "roadmap": sample_roadmap, "redirect_url": "/courses/course-python-1/roadmap"}
    create_course = AsyncMock(return_value=course)
    monkeypatch.setattr(course_api, "create_course", create_course)

    response = api_client.post(
        "/api/courses",
        json={
            "topic": "Python",
            "goal": "Learn Python fundamentals",
            "current_level": "complete_beginner",
            "pace": "medium",
            "name": "Test Student",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["course_id"] == "course-python-1"
    assert body["roadmap_ready"] is True
    assert body["roadmap"]["module_timeline"][0]["module_id"] == "M1"
    create_course.assert_awaited_once()
    assert create_course.await_args.kwargs["student_id"] == current_user["student_id"]
    assert create_course.await_args.kwargs["topic"] == "Python"
