"""Approximate queue-length estimation.

A vehicle is "queued" when its recent centre-to-centre speed falls below a
configured pixel-per-second threshold. Queue length is the polyline length of
those stopped centres, which is a pixel approximation — a later calibration
step can convert it to metres. The estimator is structured so that conversion
is a single scale factor, not a rewrite.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from smartcity_vision.analytics.trajectories import TrajectoryStore
from smartcity_vision.detection.detector import DetectionResult
from smartcity_vision.utils.config import QueueConfig
from smartcity_vision.utils.geometry import Point, euclidean_distance, polyline_length

_VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})


@dataclass(frozen=True, slots=True)
class QueueReading:
    """Queue snapshot for one frame.

    Attributes:
        queued_vehicles: Vehicles currently classified as stopped.
        length_px: Polyline length through queued centres, in pixels.
        length_m: ``length_px * metres_per_pixel`` when calibration is set.
        rolling_average_px: Mean pixel length over the configured window.
        calibrated: Whether a metres-per-pixel scale was supplied.
    """

    queued_vehicles: int
    length_px: float
    length_m: float | None
    rolling_average_px: float
    calibrated: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable reading."""
        return {
            "queued_vehicles": self.queued_vehicles,
            "length_px": round(self.length_px, 2),
            "length_m": None if self.length_m is None else round(self.length_m, 2),
            "rolling_average_px": round(self.rolling_average_px, 2),
            "calibrated": self.calibrated,
        }


class QueueEstimator:
    """Estimates how many vehicles are queued and how long the queue is."""

    def __init__(self, config: QueueConfig) -> None:
        """Initialise the estimator.

        Args:
            config: Validated queue configuration.
        """
        self._config = config
        self._window: deque[float] = deque(maxlen=config.window_frames)
        self._current: QueueReading | None = None

    def update(self, result: DetectionResult, trajectories: TrajectoryStore) -> QueueReading:
        """Classify stopped vehicles and measure the queue.

        Args:
            result: Tracked detections for one frame.
            trajectories: Centre histories used to estimate instantaneous speed.

        Returns:
            The queue reading for this frame.
        """
        queued: list[Point] = []
        for detection in result.tracked:
            if detection.class_name not in _VEHICLE_CLASSES or detection.track_id is None:
                continue
            trail = trajectories.trail(detection.track_id)
            if trail is None or len(trail.points) < 2:
                continue
            speed = _instant_speed_px_s(trail.points[-2], trail.points[-1])
            if speed <= self._config.stopped_px_per_sec:
                queued.append(detection.center)

        # Sort along x so the polyline approximates a line of vehicles rather
        # than jumping between unrelated stopped objects.
        queued.sort(key=lambda point: point[0])
        length_px = polyline_length(queued)
        self._window.append(length_px)
        average = sum(self._window) / len(self._window)
        scale = self._config.metres_per_pixel
        reading = QueueReading(
            queued_vehicles=len(queued),
            length_px=length_px,
            length_m=None if scale is None else length_px * scale,
            rolling_average_px=average,
            calibrated=scale is not None,
        )
        self._current = reading
        return reading

    @property
    def current(self) -> QueueReading | None:
        """Most recent reading, or ``None`` before the first frame."""
        return self._current

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        current = self._current
        return {"current": None if current is None else current.as_dict()}


def _instant_speed_px_s(
    previous: tuple[int, float, Point], current: tuple[int, float, Point]
) -> float:
    """Centre-to-centre speed in pixels per second."""
    dt = current[1] - previous[1]
    if dt <= 0:
        return 0.0
    return euclidean_distance(previous[2], current[2]) / dt


__all__ = ["QueueEstimator", "QueueReading"]
