"""Prometheus metrics for the API and the inference pipeline.

Counters and histograms live in one module so a scrape of
``GET /metrics/prometheus`` and an in-process ``POST /analyze/video`` update
the same instruments.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "smartcity_requests_total",
    "HTTP requests handled by the SmartCity Vision API",
    ["method", "path", "status"],
    registry=REGISTRY,
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "smartcity_inference_latency_seconds",
    "Per-frame model inference latency",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    registry=REGISTRY,
)

PIPELINE_FPS = Histogram(
    "smartcity_pipeline_fps",
    "End-to-end pipeline throughput for a completed run",
    buckets=(5, 10, 15, 20, 30, 45, 60, 90),
    registry=REGISTRY,
)

DETECTIONS_TOTAL = Counter(
    "smartcity_detections_total",
    "Detections produced by inference, labelled by class",
    ["class_name"],
    registry=REGISTRY,
)

ANALYZE_RUNS_TOTAL = Counter(
    "smartcity_analyze_runs_total",
    "Completed video-analysis runs",
    ["status"],
    registry=REGISTRY,
)


def record_request(method: str, path: str, status: int) -> None:
    """Increment the request counter for one completed HTTP call."""
    REQUESTS_TOTAL.labels(method=method, path=path, status=str(status)).inc()


def record_run(
    inference_ms_samples: list[float], fps: float, detections_by_class: dict[str, int]
) -> None:
    """Record one finished analysis run into the Prometheus instruments."""
    for sample_ms in inference_ms_samples:
        INFERENCE_LATENCY_SECONDS.observe(sample_ms / 1000.0)
    if fps > 0:
        PIPELINE_FPS.observe(fps)
    for class_name, count in detections_by_class.items():
        DETECTIONS_TOTAL.labels(class_name=class_name).inc(count)
    ANALYZE_RUNS_TOTAL.labels(status="ok").inc()


def prometheus_payload() -> tuple[bytes, str]:
    """Return the scrape body and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


__all__ = [
    "REGISTRY",
    "prometheus_payload",
    "record_request",
    "record_run",
]
