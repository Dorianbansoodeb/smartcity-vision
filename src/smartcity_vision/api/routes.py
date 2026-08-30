"""HTTP routes. Inference stays in ``cli.run``; this module only orchestrates it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Response

from smartcity_vision import __version__
from smartcity_vision.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    MetricsResponse,
    ReportExportResponse,
    TrafficCurrentResponse,
    VehicleCountsResponse,
)
from smartcity_vision.cli import run as run_analysis
from smartcity_vision.database.repository import AnalyticsRepository, resolve_database_path
from smartcity_vision.monitoring.metrics import prometheus_payload, record_run
from smartcity_vision.reports.analytics import build_summary
from smartcity_vision.reports.exporter import export_run
from smartcity_vision.utils.config import AppConfig, load_config

router = APIRouter()


def _config() -> AppConfig:
    return load_config()


def _repository(config: AppConfig | None = None) -> AnalyticsRepository:
    return AnalyticsRepository(resolve_database_path(config or _config()))


def _require_run_id(repository: AnalyticsRepository) -> str:
    run_id = repository.latest_run_id()
    if run_id is None:
        raise HTTPException(status_code=404, detail="No analysis runs have been persisted yet")
    return run_id


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Process liveness."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """High-level metrics from the latest persisted run."""
    config = _config()
    summary_path = Path(config.output.directory) / "run_summary.json"
    if not summary_path.is_file():
        return MetricsResponse()
    import json

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    stats = payload.get("stats", {})
    return MetricsResponse(
        run_id=stats.get("run_id"),
        frames_processed=int(stats.get("frames_processed") or 0),
        total_detections=int(stats.get("total_detections") or 0),
        pipeline_fps=float(stats.get("pipeline_fps") or 0.0),
        avg_inference_ms=float(stats.get("avg_inference_ms") or 0.0),
        device=str(stats.get("device") or "unknown"),
        unique_counts=stats.get("unique_counts"),
        density=stats.get("density"),
        speed=stats.get("speed"),
    )


@router.get("/metrics/prometheus")
def prometheus_metrics() -> Response:
    """Prometheus text exposition of inference and request instruments."""
    body, content_type = prometheus_payload()
    return Response(content=body, media_type=content_type)


@router.get("/vehicles/counts", response_model=VehicleCountsResponse)
def vehicle_counts() -> VehicleCountsResponse:
    """Unique object counts from the latest run summary."""
    payload = metrics()
    counts = (payload.unique_counts or {}).get("counts_by_class", {})
    return VehicleCountsResponse(
        run_id=payload.run_id,
        counts_by_class=counts,
        total=int((payload.unique_counts or {}).get("total") or 0),
    )


@router.get("/traffic/current", response_model=TrafficCurrentResponse)
def traffic_current() -> TrafficCurrentResponse:
    """Last row of ``traffic_metrics`` for the latest run."""
    repository = _repository()
    run_id = _require_run_id(repository)
    rows = repository.fetch_table("traffic_metrics", run_id)
    if not rows:
        return TrafficCurrentResponse(run_id=run_id)
    latest = rows[-1]
    return TrafficCurrentResponse(
        run_id=run_id,
        frame_index=latest["frame_index"],
        timestamp=latest["timestamp"],
        vehicles_in_frame=latest["vehicles_in_frame"],
        vehicles_in_region=latest["vehicles_in_region"],
        congestion=latest["congestion"],
        queued_vehicles=latest["queued_vehicles"],
        queue_length_px=latest["queue_length_px"],
        mean_speed_px_s=latest["mean_speed_px_s"],
    )


@router.get("/traffic/history")
def traffic_history() -> list[dict[str, Any]]:
    """Full ``traffic_metrics`` series for the latest run."""
    repository = _repository()
    run_id = _require_run_id(repository)
    return repository.fetch_table("traffic_metrics", run_id)


@router.get("/crossings")
def crossings() -> list[dict[str, Any]]:
    """Line-crossing events for the latest run."""
    repository = _repository()
    return repository.fetch_table("crossing_events", _require_run_id(repository))


@router.get("/zones/events")
def zone_events() -> list[dict[str, Any]]:
    """Zone enter/exit events for the latest run."""
    repository = _repository()
    return repository.fetch_table("zone_events", _require_run_id(repository))


@router.post("/analyze/video", response_model=AnalyzeResponse)
def analyze_video(body: AnalyzeRequest) -> AnalyzeResponse:
    """Run the analysis pipeline on ``body.source`` and persist the result."""
    updates: dict[str, Any] = {
        "video": {"source": body.source, "max_frames": body.max_frames},
        "output": {"write_annotated_video": body.write_video},
    }
    config = load_config(updates=updates)
    stats = run_analysis(config)
    record_run(stats.inference_ms_samples, stats.pipeline_fps, stats.detections_by_class)
    return AnalyzeResponse(
        run_id=stats.run_id,
        frames_processed=stats.frames_processed,
        total_detections=stats.total_detections,
        unique_counts=None if stats.counting is None else stats.counting.as_dict(),
        pipeline_fps=round(stats.pipeline_fps, 2),
        avg_inference_ms=round(stats.avg_inference_ms, 2),
        device=stats.device,
    )


@router.get("/reports/summary")
def reports_summary() -> dict[str, Any]:
    """Pandas analytics summary of the latest run."""
    config = _config()
    repository = _repository(config)
    return build_summary(repository, _require_run_id(repository))


@router.get("/reports/export", response_model=ReportExportResponse)
def reports_export() -> ReportExportResponse:
    """Re-export CSV/JSON artefacts for the latest run."""
    config = _config()
    repository = _repository(config)
    run_id = _require_run_id(repository)
    written = export_run(repository, run_id, config.output.directory)
    return ReportExportResponse(
        run_id=run_id, files={name: str(path) for name, path in written.items()}
    )
