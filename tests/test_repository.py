"""Behaviour of SQLite persistence and the pandas summary."""

from __future__ import annotations

from pathlib import Path

from smartcity_vision.analytics.line_crossing import CrossingEvent
from smartcity_vision.database.repository import AnalyticsRepository, DetectionRow, MetricRow
from smartcity_vision.detection.detector import Detection
from smartcity_vision.reports.analytics import build_summary
from smartcity_vision.reports.exporter import export_run
from smartcity_vision.utils.config import AppConfig


def test_a_run_round_trips_through_sqlite_and_the_summary(tmp_path: Path) -> None:
    repository = AnalyticsRepository(tmp_path / "test.db")
    detection = Detection(
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox=(1.0, 2.0, 3.0, 4.0),
        track_id=7,
    )
    crossing = CrossingEvent(
        line_name="mid",
        track_id=7,
        class_name="car",
        direction="A->B",
        frame_index=2,
        timestamp=0.2,
        position=(10.0, 20.0),
    )
    metric = MetricRow(
        frame_index=2,
        timestamp=0.2,
        vehicles_in_frame=1,
        vehicles_in_region=1,
        congestion="LOW",
        queued_vehicles=0,
        queue_length_px=0.0,
        mean_speed_px_s=12.5,
    )

    run_id = repository.save_run(
        AppConfig(),
        device="cpu",
        tracker="bytetrack.yaml",
        detections=[DetectionRow(frame_index=2, timestamp=0.2, detection=detection)],
        crossings=[crossing],
        zone_events=[],
        metrics=[metric],
    )

    audit = repository.fetch_audit(run_id)
    assert audit is not None
    assert audit["device"] == "cpu"
    assert audit["detections_from_id"] == audit["detections_to_id"]
    assert repository.fetch_table("detections", run_id)[0]["class_name"] == "car"
    assert repository.fetch_table("crossing_events", run_id)[0]["direction"] == "A->B"

    summary = build_summary(repository, run_id)
    assert summary["crossing_rows"] == 1
    assert summary["line_crossings"]["total"] == 1
    assert summary["average_estimated_speed_px_s"] == 12.5
    assert summary["vehicles_by_class"] == {"car": 1}

    written = export_run(repository, run_id, tmp_path / "out")
    assert (tmp_path / "out" / "events.csv").is_file()
    assert (tmp_path / "out" / "summary.json").is_file()
    assert set(written) >= {"events.csv", "traffic_metrics.csv", "summary.json"}
