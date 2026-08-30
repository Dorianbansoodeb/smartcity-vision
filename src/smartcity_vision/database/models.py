"""SQLite schema.

Tables are created with ``CREATE TABLE IF NOT EXISTS`` so a fresh database and
an existing one are both valid. The ``model_predictions_audit`` table is the
run-level provenance record: model identity, config snapshot, input source,
and a pointer to the detections that run produced.
"""

from __future__ import annotations

SCHEMA_STATEMENTS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS model_predictions_audit (
        run_id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL,
        input_source TEXT NOT NULL,
        model_weights TEXT NOT NULL,
        model_hash TEXT,
        device TEXT NOT NULL,
        tracker TEXT,
        config_json TEXT NOT NULL,
        detections_from_id INTEGER,
        detections_to_id INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        track_id INTEGER,
        class_name TEXT NOT NULL,
        confidence REAL NOT NULL,
        x1 REAL NOT NULL,
        y1 REAL NOT NULL,
        x2 REAL NOT NULL,
        y2 REAL NOT NULL,
        FOREIGN KEY (run_id) REFERENCES model_predictions_audit(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS crossing_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        line_name TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        direction TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        FOREIGN KEY (run_id) REFERENCES model_predictions_audit(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS zone_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        zone_name TEXT NOT NULL,
        zone_kind TEXT NOT NULL,
        track_id INTEGER NOT NULL,
        class_name TEXT NOT NULL,
        kind TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        dwell_seconds REAL NOT NULL,
        x REAL NOT NULL,
        y REAL NOT NULL,
        FOREIGN KEY (run_id) REFERENCES model_predictions_audit(run_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS traffic_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL,
        frame_index INTEGER NOT NULL,
        timestamp REAL NOT NULL,
        vehicles_in_frame INTEGER NOT NULL,
        vehicles_in_region INTEGER NOT NULL,
        congestion TEXT NOT NULL,
        queued_vehicles INTEGER NOT NULL,
        queue_length_px REAL NOT NULL,
        mean_speed_px_s REAL,
        FOREIGN KEY (run_id) REFERENCES model_predictions_audit(run_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_detections_run ON detections(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_crossings_run ON crossing_events(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_zones_run ON zone_events(run_id)",
    "CREATE INDEX IF NOT EXISTS idx_metrics_run ON traffic_metrics(run_id)",
)

__all__ = ["SCHEMA_STATEMENTS"]
