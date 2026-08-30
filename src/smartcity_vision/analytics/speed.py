"""Approximate vehicle speed.

The default estimate is perspective-independent pixels per second, smoothed
over a rolling window of each track's recent centres. When two reference
points and a known real-world distance are configured, the same motion is
also reported in km/h. Calibrated and uncalibrated values are never mixed:
a reading always says which one it is.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from smartcity_vision.analytics.trajectories import TrajectoryStore
from smartcity_vision.detection.detector import DetectionResult
from smartcity_vision.utils.config import SpeedConfig
from smartcity_vision.utils.geometry import euclidean_distance

_VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
_SECONDS_PER_HOUR = 3600.0
_METRES_PER_KM = 1000.0


@dataclass(frozen=True, slots=True)
class SpeedReading:
    """Smoothed speed for one track on one frame.

    Attributes:
        track_id: Identity of the object.
        class_name: Class at this frame.
        speed_px_s: Smoothed pixels per second.
        speed_kmh: Calibrated km/h, or ``None`` when uncalibrated.
        calibrated: Whether a metres-per-pixel scale was applied.
    """

    track_id: int
    class_name: str
    speed_px_s: float
    speed_kmh: float | None
    calibrated: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable reading."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "speed_px_s": round(self.speed_px_s, 2),
            "speed_kmh": None if self.speed_kmh is None else round(self.speed_kmh, 2),
            "calibrated": self.calibrated,
        }


class SpeedEstimator:
    """Per-track smoothed speed, in px/s and optionally km/h."""

    def __init__(self, config: SpeedConfig) -> None:
        """Initialise the estimator.

        Args:
            config: Validated speed configuration.
        """
        self._config = config
        self._scale = _metres_per_pixel(config)
        self._windows: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=config.window_frames)
        )
        self._current: dict[int, SpeedReading] = {}
        self._samples: list[float] = []

    def update(
        self, result: DetectionResult, trajectories: TrajectoryStore
    ) -> tuple[SpeedReading, ...]:
        """Update per-track speeds from the latest centres.

        Args:
            result: Tracked detections for one frame.
            trajectories: Centre histories used to measure motion.

        Returns:
            One reading per tracked vehicle that has enough history.
        """
        current: dict[int, SpeedReading] = {}
        for detection in result.tracked:
            if detection.class_name not in _VEHICLE_CLASSES or detection.track_id is None:
                continue
            trail = trajectories.trail(detection.track_id)
            if trail is None or len(trail.points) < 2:
                continue
            instant = _step_speed_px_s(trail.points[-2], trail.points[-1])
            window = self._windows[detection.track_id]
            window.append(instant)
            smoothed = sum(window) / len(window)
            kmh = None if self._scale is None else _px_s_to_kmh(smoothed, self._scale)
            reading = SpeedReading(
                track_id=detection.track_id,
                class_name=detection.class_name,
                speed_px_s=smoothed,
                speed_kmh=kmh,
                calibrated=self._scale is not None,
            )
            current[detection.track_id] = reading
            self._samples.append(smoothed)
        self._current = current
        return tuple(current[track_id] for track_id in sorted(current))

    @property
    def current(self) -> tuple[SpeedReading, ...]:
        """Readings for vehicles present on the latest frame."""
        return tuple(self._current[track_id] for track_id in sorted(self._current))

    @property
    def calibrated(self) -> bool:
        """Whether this estimator converts to km/h."""
        return self._scale is not None

    def mean_speed_px_s(self) -> float | None:
        """Mean of every smoothed sample this run, or ``None`` if none yet."""
        if not self._samples:
            return None
        return sum(self._samples) / len(self._samples)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        mean = self.mean_speed_px_s()
        mean_kmh = None if mean is None or self._scale is None else _px_s_to_kmh(mean, self._scale)
        return {
            "calibrated": self.calibrated,
            "mean_speed_px_s": None if mean is None else round(mean, 2),
            "mean_speed_kmh": None if mean_kmh is None else round(mean_kmh, 2),
            "active": [reading.as_dict() for reading in self.current],
        }


def _metres_per_pixel(config: SpeedConfig) -> float | None:
    """Resolve a metres-per-pixel scale, or ``None`` when uncalibrated."""
    if config.metres_per_pixel is not None:
        return config.metres_per_pixel
    if config.reference_points is None or config.reference_distance_m is None:
        return None
    first, second = config.reference_points
    pixels = euclidean_distance(first, second)
    if pixels <= 0:
        return None
    return config.reference_distance_m / pixels


def _step_speed_px_s(
    previous: tuple[int, float, tuple[float, float]],
    current: tuple[int, float, tuple[float, float]],
) -> float:
    """Instantaneous centre-to-centre speed in pixels per second."""
    dt = current[1] - previous[1]
    if dt <= 0:
        return 0.0
    return euclidean_distance(previous[2], current[2]) / dt


def _px_s_to_kmh(speed_px_s: float, metres_per_pixel: float) -> float:
    """Convert pixels/second to km/h using a uniform scale."""
    return speed_px_s * metres_per_pixel * _SECONDS_PER_HOUR / _METRES_PER_KM


__all__ = ["SpeedEstimator", "SpeedReading"]
