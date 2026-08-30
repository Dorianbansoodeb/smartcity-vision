"""Behaviour of the FastAPI surface that does not require a model."""

from __future__ import annotations

from fastapi.testclient import TestClient

from smartcity_vision import __version__
from smartcity_vision.api.app import create_app


def test_health_and_prometheus_are_reachable() -> None:
    client = TestClient(create_app())

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "version": __version__}

    scrape = client.get("/metrics/prometheus")
    assert scrape.status_code == 200
    body = scrape.text
    assert "smartcity_requests_total" in body


def test_missing_run_returns_404_rather_than_an_empty_invention(tmp_path, monkeypatch) -> None:
    from smartcity_vision.api import routes
    from smartcity_vision.database.repository import AnalyticsRepository
    from smartcity_vision.utils.config import AppConfig

    def fake_config(**_kwargs):
        return AppConfig.model_validate({"output": {"directory": tmp_path}})

    monkeypatch.setattr(routes, "load_config", fake_config)
    AnalyticsRepository(tmp_path / "smartcity.db")
    client = TestClient(create_app())

    response = client.get("/crossings")
    assert response.status_code == 404
    assert "No analysis runs" in response.json()["detail"]
