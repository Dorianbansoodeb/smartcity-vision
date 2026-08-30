#!/usr/bin/env python3
"""Export the latest (or a named) run to CSV and JSON.

Example:
    python scripts/export_report.py
    python scripts/export_report.py --run-id <hex> --output-dir data/output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartcity_vision.database.repository import AnalyticsRepository  # noqa: E402
from smartcity_vision.exceptions import SmartCityVisionError  # noqa: E402
from smartcity_vision.reports.exporter import export_run  # noqa: E402
from smartcity_vision.utils.config import load_config  # noqa: E402
from smartcity_vision.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Export a persisted run and return a process exit code."""
    parser = argparse.ArgumentParser(description="Export SmartCity Vision run reports.")
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path")
    parser.add_argument("--run-id", default=None, help="Run to export (default: latest)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination directory")
    parser.add_argument("--config", "-c", type=Path, default=None)
    args = parser.parse_args(argv)

    setup_logging("INFO")
    try:
        config = load_config(args.config)
        db_path = args.db or (config.output.directory / "smartcity.db")
        output_dir = args.output_dir or config.output.directory
        if not db_path.is_file():
            raise SmartCityVisionError(f"Database not found: {db_path}")
        repository = AnalyticsRepository(db_path)
        run_id = args.run_id or repository.latest_run_id()
        if run_id is None:
            raise SmartCityVisionError(f"No runs stored in {db_path}")
        written = export_run(repository, run_id, output_dir)
    except SmartCityVisionError as exc:
        logger.error("%s", exc)
        return 1

    for name, path in written.items():
        logger.info("Wrote %s -> %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
