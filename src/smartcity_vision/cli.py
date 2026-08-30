"""Command-line entry point.

Flags are deliberately thin wrappers over config keys: anything settable on the
command line is also settable in YAML, and ``--set section.key=value`` reaches
options without a dedicated flag. The scripts in ``scripts/`` delegate here so
there is one implementation of the run sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from smartcity_vision import __version__
from smartcity_vision.database.repository import AnalyticsRepository, resolve_database_path
from smartcity_vision.detection.detector import YoloDetector
from smartcity_vision.detection.tracker import YoloTracker
from smartcity_vision.exceptions import SmartCityVisionError
from smartcity_vision.experiments import log_run
from smartcity_vision.reports.exporter import export_run
from smartcity_vision.utils.config import AppConfig, load_config
from smartcity_vision.utils.logging import get_logger, setup_logging
from smartcity_vision.video.processor import (
    ProcessingStats,
    VideoProcessor,
    create_source_from_config,
)

logger = get_logger(__name__)

SUMMARY_FILENAME = "run_summary.json"


def build_parser(description: str, default_source: str | None = None) -> argparse.ArgumentParser:
    """Build the shared argument parser.

    Args:
        description: Help text shown for the command.
        default_source: Value used for ``--input`` when the flag is omitted, e.g.
            ``"0"`` for the webcam script. ``None`` leaves the config value in place.

    Returns:
        A configured parser.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--version", action="version", version=f"SmartCity Vision {__version__}")

    source_help = "Video file path, webcam index (e.g. 0), or stream URL (rtsp://...)"
    parser.add_argument("--input", "-i", default=default_source, help=source_help)
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="YAML config file (default: config/default.yaml when present)",
    )
    parser.add_argument("--weights", type=Path, default=None, help="Path to YOLO weights")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Inference device (default: auto-detect)",
    )
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    parser.add_argument(
        "--tracker",
        choices=["bytetrack.yaml", "botsort.yaml"],
        default=None,
        help="Tracking algorithm (default: bytetrack.yaml)",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Run stateless detection only; disables unique object counting",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=None,
        help="Frames to drop between processed frames (0 = process every frame)",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--loop", action="store_true", help="Restart file sources at end of file")
    parser.add_argument("--display", action="store_true", help="Show a live preview window")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for artefacts")
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip writing the annotated video (metrics only)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Logging verbosity",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Override any config key, e.g. --set visualization.show_hud=false",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    """Turn parsed arguments into a validated :class:`AppConfig`."""
    model: dict[str, Any] = {}
    tracking: dict[str, Any] = {}
    video: dict[str, Any] = {}
    output: dict[str, Any] = {}
    logging_section: dict[str, Any] = {}

    if args.weights is not None:
        model["weights"] = args.weights
    if args.device is not None:
        model["device"] = args.device
    if args.conf is not None:
        model["confidence_threshold"] = args.conf
    if args.iou is not None:
        model["iou_threshold"] = args.iou
    if args.imgsz is not None:
        model["image_size"] = args.imgsz

    if args.tracker is not None:
        tracking["tracker"] = args.tracker
    if args.no_tracking:
        tracking["enabled"] = False

    if args.input is not None:
        video["source"] = str(args.input)
    if args.frame_skip is not None:
        video["frame_skip"] = args.frame_skip
    if args.max_frames is not None:
        video["max_frames"] = args.max_frames
    if args.loop:
        video["loop"] = True
    if args.display:
        video["display"] = True

    if args.output_dir is not None:
        output["directory"] = args.output_dir
    if args.no_video:
        output["write_annotated_video"] = False

    if args.log_level is not None:
        logging_section["level"] = args.log_level

    updates = {
        section: values
        for section, values in (
            ("model", model),
            ("tracking", tracking),
            ("video", video),
            ("output", output),
            ("logging", logging_section),
        )
        if values
    }
    return load_config(path=args.config, overrides=args.overrides, updates=updates)


def run(config: AppConfig) -> ProcessingStats:
    """Execute one analysis run and persist its measured summary.

    Args:
        config: Validated configuration.

    Returns:
        Statistics measured during the run.
    """
    # Build the source first: a bad path or unreachable stream should fail before
    # spending seconds loading model weights.
    source = create_source_from_config(config)
    detector = (
        YoloTracker(config.model, config.tracking)
        if config.tracking.enabled
        else YoloDetector(config.model)
    )
    stats = VideoProcessor(config, detector, source=source).run()
    _persist_run(config, stats)
    _write_summary(config, stats)
    log_run(config, stats)
    return stats


def main(argv: list[str] | None = None, default_source: str | None = None) -> int:
    """Parse arguments, run the pipeline, and return a process exit code.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.
        default_source: Default value for ``--input``.

    Returns:
        ``0`` on success, ``1`` on a handled SmartCity Vision error, ``130`` if
        interrupted.
    """
    parser = build_parser(
        description="Analyse traffic video with YOLOv8 and write an annotated result.",
        default_source=default_source,
    )
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
    except SmartCityVisionError as exc:
        setup_logging("INFO")
        logger.error("%s", exc)
        return 1

    setup_logging(config.logging.level, config.logging.file)
    logger.debug("Effective config: %s", json.dumps(config.snapshot(), indent=2))

    try:
        stats = run(config)
    except SmartCityVisionError as exc:
        logger.error("%s", exc)
        return 1

    return 130 if stats.interrupted else 0


def _persist_run(config: AppConfig, stats: ProcessingStats) -> None:
    """Write the run to SQLite and export CSV/JSON artefacts."""
    repository = AnalyticsRepository(resolve_database_path(config))
    crossings = () if stats.crossings is None else stats.crossings.events
    zones = () if stats.zones is None else stats.zones.events
    run_id = repository.save_run(
        config,
        stats.device,
        stats.tracker,
        stats.detection_rows,
        crossings,
        zones,
        stats.metric_rows,
    )
    stats.run_id = run_id
    export_run(repository, run_id, config.output.directory)


def _write_summary(config: AppConfig, stats: ProcessingStats) -> None:
    """Write the run summary, including the config snapshot, as JSON."""
    directory = config.output.directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SUMMARY_FILENAME
    payload = {"config": config.snapshot(), "stats": stats.as_dict()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote run summary to %s", path)


__all__ = ["build_parser", "config_from_args", "main", "run"]
