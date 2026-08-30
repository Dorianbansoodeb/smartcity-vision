"""Pydantic response models for the FastAPI surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness payload."""

    status: str = "ok"
    version: str


class VehicleCountsResponse(BaseModel):
    """Unique object counts from the latest run."""

    run_id: str | None = None
    counts_by_class: dict[str, int] = Field(default_factory=dict)
    total: int = 0


class TrafficCurrentResponse(BaseModel):
    """Latest per-frame traffic snapshot."""

    run_id: str | None = None
    frame_index: int | None = None
    timestamp: float | None = None
    vehicles_in_frame: int = 0
    vehicles_in_region: int = 0
    congestion: str = "LOW"
    queued_vehicles: int = 0
    queue_length_px: float = 0.0
    mean_speed_px_s: float | None = None


class AnalyzeRequest(BaseModel):
    """Body for ``POST /analyze/video``."""

    source: str
    max_frames: int | None = Field(default=None, ge=1)
    write_video: bool = False


class AnalyzeResponse(BaseModel):
    """Result of a completed analysis run."""

    run_id: str | None
    frames_processed: int
    total_detections: int
    unique_counts: dict[str, Any] | None
    pipeline_fps: float
    avg_inference_ms: float
    device: str


class ReportExportResponse(BaseModel):
    """Paths written by a report export."""

    run_id: str
    files: dict[str, str]


class MetricsResponse(BaseModel):
    """High-level metrics for the latest run."""

    run_id: str | None = None
    frames_processed: int = 0
    total_detections: int = 0
    pipeline_fps: float = 0.0
    avg_inference_ms: float = 0.0
    device: str = "unknown"
    unique_counts: dict[str, Any] | None = None
    density: dict[str, Any] | None = None
    speed: dict[str, Any] | None = None
