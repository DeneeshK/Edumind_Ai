from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_health_endpoint_is_lightweight(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint_checks_database_with_mocked_db(api_client, monkeypatch):
    import db.postgres as postgres

    class FakePool:
        async def fetchval(self, query):
            assert query == "SELECT 1"
            return 1

    async def fake_get_pool():
        return FakePool()

    monkeypatch.setattr(postgres, "get_pool", fake_get_pool)

    response = api_client.get("/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["db"] == "ok"
