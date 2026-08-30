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


def test_demo_page_and_idle_status_are_reachable() -> None:
    client = TestClient(create_app())

    page = client.get("/")
    assert page.status_code == 200
    assert "Run it yourself" in page.text
    preview = client.get("/demo/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("image/gif")

    status = client.get("/demo/status")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "idle"
    assert body["max_frames"] == 64
    assert body["stats"] is None


def test_demo_analyze_refuses_when_the_sample_clip_is_missing(monkeypatch, tmp_path) -> None:
    from smartcity_vision.api import demo

    monkeypatch.setattr(demo, "SAMPLE_VIDEO", tmp_path / "missing.mp4")
    client = TestClient(create_app())

    response = client.post("/demo/analyze")
    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


def test_demo_analyze_refuses_a_second_job(monkeypatch, tmp_path) -> None:
    from smartcity_vision.api import demo

    sample = tmp_path / "traffic.mp4"
    sample.write_bytes(b"not-a-real-video")
    monkeypatch.setattr(demo, "SAMPLE_VIDEO", sample)
    client = TestClient(create_app())
    assert demo._job_lock.acquire(blocking=False)
    try:
        response = client.post("/demo/analyze")
        assert response.status_code == 409
        assert "already running" in response.json()["detail"]
    finally:
        demo._job_lock.release()


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
