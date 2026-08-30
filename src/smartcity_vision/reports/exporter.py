"""Write analytics artefacts to CSV and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from smartcity_vision.database.repository import AnalyticsRepository
from smartcity_vision.reports.analytics import build_summary
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)


def export_run(
    repository: AnalyticsRepository,
    run_id: str,
    directory: Path,
) -> dict[str, Path]:
    """Export one run to CSV tables plus a JSON summary.

    Args:
        repository: Open analytics database.
        run_id: Persisted run identifier.
        directory: Destination directory, created if needed.

    Returns:
        Mapping of artefact name to the path that was written.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for table, filename in (
        ("detections", "events.csv"),
        ("crossing_events", "crossing_events.csv"),
        ("zone_events", "zone_events.csv"),
        ("traffic_metrics", "traffic_metrics.csv"),
    ):
        frame = pd.DataFrame(repository.fetch_table(table, run_id))
        path = directory / filename
        frame.to_csv(path, index=False)
        written[filename] = path

    summary = build_summary(repository, run_id)
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    written["summary.json"] = summary_path
    logger.info("Exported run %s to %s", run_id, directory)
    return written


def _jsonable(value: Any) -> Any:
    """Convert pandas / numpy leftovers into JSON-native types."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


__all__ = ["export_run"]
