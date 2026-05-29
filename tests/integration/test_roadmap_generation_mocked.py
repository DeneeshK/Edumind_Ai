from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import app.course_api as course_api
from app.auth import require_current_user


pytestmark = pytest.mark.integration


def test_course_roadmap_endpoint_returns_mocked_roadmap(
    api_app,
    api_client,
    current_user,
    sample_course,
    sample_roadmap,
    monkeypatch,
):
    api_app.dependency_overrides[require_current_user] = lambda: current_user
    monkeypatch.setattr(course_api, "get_course_for_student", AsyncMock(return_value=sample_course))
    monkeypatch.setattr(course_api, "get_course_roadmap", AsyncMock(return_value=sample_roadmap))

    response = api_client.get("/api/courses/course-python-1/roadmap")

    assert response.status_code == 200
    body = response.json()
    assert body["course"]["id"] == "course-python-1"
    assert [m["module_id"] for m in body["roadmap"]["module_timeline"]] == ["M1", "M2"]


def test_course_roadmap_endpoint_404_when_missing(
    api_app,
    api_client,
    current_user,
    sample_course,
    monkeypatch,
):
    api_app.dependency_overrides[require_current_user] = lambda: current_user
    monkeypatch.setattr(course_api, "get_course_for_student", AsyncMock(return_value=sample_course))
    monkeypatch.setattr(course_api, "get_course_roadmap", AsyncMock(return_value=None))

    response = api_client.get("/api/courses/course-python-1/roadmap")

    assert response.status_code == 404
