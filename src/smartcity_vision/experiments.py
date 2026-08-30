"""MLflow experiment tracking and champion/challenger comparison.

Every analysis run can be logged with its config snapshot and the metrics that
run actually produced. A champion/challenger helper runs the same clip through
two weight files and writes a side-by-side report — the framing regulated ML
teams use before promoting a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from smartcity_vision.utils.config import AppConfig
from smartcity_vision.utils.logging import get_logger
from smartcity_vision.video.processor import ProcessingStats

logger = get_logger(__name__)

EXPERIMENT_NAME = "smartcity-vision"


def log_run(config: AppConfig, stats: ProcessingStats) -> None:
    """Log one analysis run to MLflow if the library is importable.

    Failure to talk to a tracking server is a warning, not a crash: a laptop
    demo should still finish if MLflow is missing or the store is unreachable.
    """
    try:
        import mlflow
    except ImportError:
        logger.info("MLflow is not installed; skipping experiment logging")
        return

    try:
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=stats.run_id or "local"):
            mlflow.log_params(
                {
                    "weights": str(config.model.weights),
                    "device": stats.device,
                    "tracker": stats.tracker or "none",
                    "image_size": config.model.image_size,
                    "confidence_threshold": config.model.confidence_threshold,
                    "source": config.video.source,
                }
            )
            mlflow.log_metrics(
                {
                    "frames_processed": stats.frames_processed,
                    "total_detections": stats.total_detections,
                    "pipeline_fps": stats.pipeline_fps,
                    "avg_inference_ms": stats.avg_inference_ms,
                    "p95_inference_ms": stats.p95_inference_ms,
                    "unique_total": 0 if stats.counting is None else stats.counting.total,
                }
            )
            mlflow.log_dict(config.snapshot(), "config.json")
            mlflow.log_dict(stats.as_dict(), "stats.json")
    except Exception as exc:  # noqa: BLE001 — tracking must not fail the run
        logger.warning("MLflow logging failed: %s", exc)


def write_challenger_report(
    champion: dict[str, Any],
    challenger: dict[str, Any],
    destination: Path,
) -> Path:
    """Write a side-by-side comparison of two already-measured runs.

    Args:
        champion: Stats dict from the production candidate.
        challenger: Stats dict from the model under test.
        destination: JSON file to write.

    Returns:
        ``destination``.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "champion": champion,
        "challenger": challenger,
        "deltas": {
            "avg_inference_ms": _delta(champion, challenger, "avg_inference_ms"),
            "p95_inference_ms": _delta(champion, challenger, "p95_inference_ms"),
            "pipeline_fps": _delta(champion, challenger, "pipeline_fps"),
            "total_detections": _delta(champion, challenger, "total_detections"),
        },
        "note": (
            "Precision/recall/mAP are omitted because this clip has no ground-truth "
            "labels. Compare latency and detection counts only; do not treat a higher "
            "detection count as higher quality."
        ),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote champion/challenger report to %s", destination)
    return destination


def _delta(champion: dict[str, Any], challenger: dict[str, Any], key: str) -> float | None:
    """Challenger minus champion for a numeric field."""
    left, right = champion.get(key), challenger.get(key)
    if left is None or right is None:
        return None
    return round(float(right) - float(left), 3)


__all__ = ["log_run", "write_challenger_report"]
