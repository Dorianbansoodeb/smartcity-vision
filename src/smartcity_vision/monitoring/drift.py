"""Class-distribution drift detection.

Compares the class mix in a rolling window of detections against a stored
baseline. The distance is total variation (half the L1 distance between the
two probability vectors), which is 0 when the mixes match and 1 when they
share no mass. A configurable threshold flags the window.

In production this would page on-call via Prometheus Alertmanager when
``smartcity_drift_detected`` flips; this module is the check that alert
would scrape.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

from smartcity_vision.detection.detector import DetectionResult
from smartcity_vision.utils.config import MonitoringConfig
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)


def class_distribution(counts: dict[str, int]) -> dict[str, float]:
    """Normalise raw counts into a probability vector."""
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {name: count / total for name, count in counts.items()}


def total_variation(first: dict[str, float], second: dict[str, float]) -> float:
    """Total variation distance between two discrete distributions."""
    keys = set(first) | set(second)
    return 0.5 * sum(abs(first.get(key, 0.0) - second.get(key, 0.0)) for key in keys)


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Result of comparing a window against the baseline.

    Attributes:
        drifted: Whether the distance exceeds the configured threshold.
        distance: Total variation distance in ``[0, 1]``.
        threshold: The threshold that was applied.
        window_detections: How many detections the window contained.
        baseline_detections: How many detections built the baseline.
        window_distribution: Normalised class mix of the window.
    """

    drifted: bool
    distance: float
    threshold: float
    window_detections: int
    baseline_detections: int
    window_distribution: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable report."""
        return {
            "drifted": self.drifted,
            "distance": round(self.distance, 4),
            "threshold": self.threshold,
            "window_detections": self.window_detections,
            "baseline_detections": self.baseline_detections,
            "window_distribution": {
                k: round(v, 4) for k, v in sorted(self.window_distribution.items())
            },
        }


class DriftDetector:
    """Rolling-window class-mix drift check."""

    def __init__(
        self,
        config: MonitoringConfig,
        baseline: dict[str, int] | None = None,
    ) -> None:
        """Initialise the detector.

        Args:
            config: Validated monitoring configuration.
            baseline: Optional precomputed class counts. When omitted the first
                completed window becomes the baseline, so a cold start cannot
                false-alarm against an empty prior.
        """
        self._config = config
        self._window: deque[str] = deque()
        self._baseline = Counter(baseline or {})
        self._latest: DriftReport | None = None

    def observe(self, class_name: str) -> DriftReport | None:
        """Add one detection and, once the window is full, score it."""
        self._window.append(class_name)
        while len(self._window) > self._config.drift_window_frames:
            self._window.popleft()
        if len(self._window) < self._config.drift_window_frames:
            return None
        if sum(self._baseline.values()) < self._config.drift_min_detections:
            self._baseline.update(self._window)
            return None
        return self._score()

    def update(self, result: DetectionResult) -> DriftReport | None:
        """Observe every detection in ``result``; return the latest score."""
        report = None
        for detection in result.detections:
            report = self.observe(detection.class_name) or report
        return report

    def set_baseline(self, counts: dict[str, int]) -> None:
        """Replace the stored baseline, e.g. from a previous production week."""
        self._baseline = Counter(counts)

    @property
    def latest(self) -> DriftReport | None:
        """Most recent score, or ``None`` before the window fills."""
        return self._latest

    def _score(self) -> DriftReport:
        window_counts: Counter[str] = Counter(self._window)
        window_dist = class_distribution(dict(window_counts))
        baseline_dist = class_distribution(dict(self._baseline))
        distance = total_variation(window_dist, baseline_dist)
        report = DriftReport(
            drifted=distance >= self._config.drift_threshold,
            distance=distance,
            threshold=self._config.drift_threshold,
            window_detections=len(self._window),
            baseline_detections=sum(self._baseline.values()),
            window_distribution=window_dist,
        )
        self._latest = report
        if report.drifted:
            logger.warning(
                "Class-distribution drift detected: TV=%.3f (threshold %.3f) window=%s",
                report.distance,
                report.threshold,
                report.window_distribution,
            )
        return report


__all__ = ["DriftDetector", "DriftReport", "class_distribution", "total_variation"]
