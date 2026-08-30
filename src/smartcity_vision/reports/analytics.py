"""Build a pandas summary from a persisted run.

Every number here is computed from rows that were written by an actual run.
If a table is empty the corresponding section says so instead of inventing
a zero that looks like a measurement.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from smartcity_vision.database.repository import AnalyticsRepository


def build_summary(repository: AnalyticsRepository, run_id: str) -> dict[str, Any]:
    """Return a structured analytics summary for ``run_id``.

    Args:
        repository: Open analytics database.
        run_id: Persisted run identifier.

    Returns:
        Nested dictionaries ready for JSON export. Missing tables become empty
        lists rather than fabricated metrics.
    """
    audit = repository.fetch_audit(run_id)
    detections = pd.DataFrame(repository.fetch_table("detections", run_id))
    crossings = pd.DataFrame(repository.fetch_table("crossing_events", run_id))
    zones = pd.DataFrame(repository.fetch_table("zone_events", run_id))
    metrics = pd.DataFrame(repository.fetch_table("traffic_metrics", run_id))

    duration_s = _duration_seconds(detections, metrics)
    return {
        "run_id": run_id,
        "audit": audit,
        "vehicles_per_minute": _vehicles_per_minute(detections, duration_s),
        "vehicles_by_class": _value_counts(detections, "class_name"),
        "line_crossings": _crossing_summary(crossings),
        "direction_distribution": _value_counts(crossings, "direction"),
        "traffic_density_over_time": _series(metrics, "vehicles_in_region"),
        "queue_length_over_time": _series(metrics, "queue_length_px"),
        "average_estimated_speed_px_s": _mean(metrics, "mean_speed_px_s"),
        "peak_congestion_periods": _peak_congestion(metrics),
        "duration_seconds": duration_s,
        "detection_rows": int(len(detections)),
        "crossing_rows": int(len(crossings)),
        "zone_event_rows": int(len(zones)),
        "metric_rows": int(len(metrics)),
    }


def _duration_seconds(detections: pd.DataFrame, metrics: pd.DataFrame) -> float | None:
    """Elapsed time of the run from stored timestamps."""
    frames = detections if not detections.empty else metrics
    if frames.empty or "timestamp" not in frames:
        return None
    return float(frames["timestamp"].max() - frames["timestamp"].min())


def _vehicles_per_minute(detections: pd.DataFrame, duration_s: float | None) -> dict[str, Any]:
    """Detection rate, clearly labelled as detections not unique objects."""
    if detections.empty or not duration_s or duration_s <= 0:
        return {"detections_per_minute": None, "note": "not enough rows to compute a rate"}
    return {
        "detections_per_minute": round(len(detections) / duration_s * 60.0, 2),
        "note": "This is detections per minute, not unique objects.",
    }


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    """Return a sorted value-count mapping, or empty if the column is absent."""
    if frame.empty or column not in frame:
        return {}
    return {str(key): int(value) for key, value in frame[column].value_counts().items()}


def _crossing_summary(crossings: pd.DataFrame) -> dict[str, Any]:
    """Aggregate crossings by line and class."""
    if crossings.empty:
        return {"total": 0, "by_line": {}, "by_class": {}}
    return {
        "total": int(len(crossings)),
        "by_line": _value_counts(crossings, "line_name"),
        "by_class": _value_counts(crossings, "class_name"),
    }


def _series(frame: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    """Return a compact time series of ``column``."""
    if frame.empty or column not in frame:
        return []
    return [
        {
            "frame_index": _as_int(row.frame_index),
            "timestamp": _as_float(row.timestamp),
            column: None if pd.isna(getattr(row, column)) else _as_float(getattr(row, column)),
        }
        for row in frame.itertuples(index=False)
    ]


def _as_int(value: object) -> int:
    """Coerce a pandas cell that this table stores as a whole number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a numeric cell, got {type(value)!r}")
    return int(value)


def _as_float(value: object) -> float:
    """Coerce a pandas cell that this table stores as a real number."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a numeric cell, got {type(value)!r}")
    return float(value)


def _mean(frame: pd.DataFrame, column: str) -> float | None:
    """Mean of a numeric column, ignoring nulls."""
    if frame.empty or column not in frame:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return round(float(series.mean()), 2)


def _peak_congestion(metrics: pd.DataFrame) -> list[dict[str, Any]]:
    """Frames whose congestion label is the highest observed."""
    if metrics.empty or "congestion" not in metrics:
        return []
    rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
    peak = max(metrics["congestion"].map(rank).fillna(0))
    if peak <= 0:
        return []
    labels = {value: key for key, value in rank.items()}
    peak_label = labels[int(peak)]
    subset = metrics[metrics["congestion"] == peak_label]
    return [
        {
            "frame_index": _as_int(row.frame_index),
            "timestamp": _as_float(row.timestamp),
            "congestion": peak_label,
            "vehicles_in_region": _as_int(row.vehicles_in_region),
        }
        for row in subset.itertuples(index=False)
    ]


__all__ = ["build_summary"]
