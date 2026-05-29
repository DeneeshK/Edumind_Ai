from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import app.course_api as course_api
from app.auth import require_current_user


pytestmark = pytest.mark.integration


def test_module_generation_endpoint_returns_mocked_lesson(
    api_app,
    api_client,
    current_user,
    sample_course,
    monkeypatch,
):
    api_app.dependency_overrides[require_current_user] = lambda: current_user
    module = {
        "id": "M1",
        "title": "Values and Variables",
        "concept": "variables",
        "content_markdown": "# Values and Variables\n\nVariables store values.",
        "questions": [],
        "videos": [],
    }
    monkeypatch.setattr(
        course_api,
        "_require_owned_module",
        AsyncMock(return_value=(sample_course, module)),
    )
    generate_module_lesson = AsyncMock(return_value=module)
    monkeypatch.setattr(course_api, "generate_module_lesson", generate_module_lesson)

    response = api_client.post("/api/courses/course-python-1/modules/M1/generate")

    assert response.status_code == 200
    body = response.json()
    assert body["module"]["id"] == "M1"
    assert body["module"]["content_markdown"].startswith("# Values and Variables")
    generate_module_lesson.assert_awaited_once_with(
        "course-python-1",
        "M1",
        current_user["student_id"],
    )
