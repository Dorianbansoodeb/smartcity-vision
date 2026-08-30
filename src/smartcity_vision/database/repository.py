"""SQLite repository for one analysis run.

The repository owns schema creation and the write of a finished run. Reads
used by the API and the report exporter go through the same connection helper
so there is one place that knows the file path.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from smartcity_vision.analytics.line_crossing import CrossingEvent
from smartcity_vision.analytics.zones import ZoneEvent
from smartcity_vision.database.models import SCHEMA_STATEMENTS
from smartcity_vision.detection.detector import Detection
from smartcity_vision.utils.config import AppConfig
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_DB_NAME = "smartcity.db"


@dataclass(frozen=True, slots=True)
class DetectionRow:
    """One detection persisted from a run."""

    frame_index: int
    timestamp: float
    detection: Detection


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One per-frame traffic-metric snapshot."""

    frame_index: int
    timestamp: float
    vehicles_in_frame: int
    vehicles_in_region: int
    congestion: str
    queued_vehicles: int
    queue_length_px: float
    mean_speed_px_s: float | None


class AnalyticsRepository:
    """Read/write access to the SQLite analytics database."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the database at ``path`` and ensure the schema exists."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            for statement in SCHEMA_STATEMENTS:
                connection.execute(statement)
            connection.commit()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection with row access by name."""
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def save_run(
        self,
        config: AppConfig,
        device: str,
        tracker: str | None,
        detections: Sequence[DetectionRow],
        crossings: Sequence[CrossingEvent],
        zone_events: Sequence[ZoneEvent],
        metrics: Sequence[MetricRow],
    ) -> str:
        """Persist one finished run and return its ``run_id``."""
        run_id = uuid4().hex
        started_at = datetime.now(UTC).isoformat()
        weights = Path(config.model.weights)
        model_hash = _file_sha256(weights) if weights.is_file() else None

        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO model_predictions_audit (
                    run_id, started_at, input_source, model_weights, model_hash,
                    device, tracker, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    started_at,
                    config.video.source,
                    str(config.model.weights),
                    model_hash,
                    device,
                    tracker,
                    json.dumps(config.snapshot()),
                ),
            )
            detection_ids = self._insert_detections(connection, run_id, detections)
            self._insert_crossings(connection, run_id, crossings)
            self._insert_zone_events(connection, run_id, zone_events)
            self._insert_metrics(connection, run_id, metrics)
            if detection_ids:
                connection.execute(
                    """
                    UPDATE model_predictions_audit
                    SET detections_from_id = ?, detections_to_id = ?
                    WHERE run_id = ?
                    """,
                    (detection_ids[0], detection_ids[-1], run_id),
                )
            connection.commit()

        logger.info(
            "Persisted run %s (%d detections, %d crossings, %d zone events, %d metric rows) to %s",
            run_id,
            len(detections),
            len(crossings),
            len(zone_events),
            len(metrics),
            self.path,
        )
        return run_id

    def latest_run_id(self) -> str | None:
        """Return the most recently written run id, or ``None`` if the table is empty."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT run_id FROM model_predictions_audit ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
        return None if row is None else str(row["run_id"])

    def fetch_audit(self, run_id: str) -> dict[str, Any] | None:
        """Return the audit row for ``run_id``."""
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_predictions_audit WHERE run_id = ?", (run_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def fetch_table(self, table: str, run_id: str | None = None) -> list[dict[str, Any]]:
        """Return rows from ``table``, optionally filtered to one run."""
        allowed = {
            "detections",
            "crossing_events",
            "zone_events",
            "traffic_metrics",
            "model_predictions_audit",
        }
        if table not in allowed:
            raise ValueError(f"Unknown table {table!r}")
        query = f"SELECT * FROM {table}"  # noqa: S608 — table name is allow-listed
        params: tuple[Any, ...] = ()
        if run_id is not None:
            query += " WHERE run_id = ?"
            params = (run_id,)
        with self.connect() as connection:
            return [dict(row) for row in connection.execute(query, params)]

    @staticmethod
    def _insert_detections(
        connection: sqlite3.Connection, run_id: str, rows: Sequence[DetectionRow]
    ) -> list[int]:
        ids: list[int] = []
        for row in rows:
            cursor = connection.execute(
                """
                INSERT INTO detections (
                    run_id, frame_index, timestamp, track_id, class_name,
                    confidence, x1, y1, x2, y2
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    row.frame_index,
                    row.timestamp,
                    row.detection.track_id,
                    row.detection.class_name,
                    row.detection.confidence,
                    *row.detection.bbox,
                ),
            )
            if cursor.lastrowid is not None:
                ids.append(int(cursor.lastrowid))
        return ids

    @staticmethod
    def _insert_crossings(
        connection: sqlite3.Connection, run_id: str, events: Sequence[CrossingEvent]
    ) -> None:
        connection.executemany(
            """
            INSERT INTO crossing_events (
                run_id, line_name, track_id, class_name, direction,
                frame_index, timestamp, x, y
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    event.line_name,
                    event.track_id,
                    event.class_name,
                    event.direction,
                    event.frame_index,
                    event.timestamp,
                    event.position[0],
                    event.position[1],
                )
                for event in events
            ],
        )

    @staticmethod
    def _insert_zone_events(
        connection: sqlite3.Connection, run_id: str, events: Sequence[ZoneEvent]
    ) -> None:
        connection.executemany(
            """
            INSERT INTO zone_events (
                run_id, zone_name, zone_kind, track_id, class_name, kind,
                frame_index, timestamp, dwell_seconds, x, y
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    event.zone_name,
                    event.zone_kind,
                    event.track_id,
                    event.class_name,
                    event.kind,
                    event.frame_index,
                    event.timestamp,
                    event.dwell_seconds,
                    event.position[0],
                    event.position[1],
                )
                for event in events
            ],
        )

    @staticmethod
    def _insert_metrics(
        connection: sqlite3.Connection, run_id: str, rows: Sequence[MetricRow]
    ) -> None:
        connection.executemany(
            """
            INSERT INTO traffic_metrics (
                run_id, frame_index, timestamp, vehicles_in_frame, vehicles_in_region,
                congestion, queued_vehicles, queue_length_px, mean_speed_px_s
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    row.frame_index,
                    row.timestamp,
                    row.vehicles_in_frame,
                    row.vehicles_in_region,
                    row.congestion,
                    row.queued_vehicles,
                    row.queue_length_px,
                    row.mean_speed_px_s,
                )
                for row in rows
            ],
        )


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_database_path(config: AppConfig) -> Path:
    """Return the SQLite path for a run, under the configured output directory."""
    return Path(config.output.directory) / DEFAULT_DB_NAME


__all__ = [
    "AnalyticsRepository",
    "DetectionRow",
    "MetricRow",
    "resolve_database_path",
]
