"""Traffic density and congestion classification.

Density is the number of active tracked vehicles (not pedestrians) in a region,
plus a rolling average so a single crowded frame cannot flip the congestion
label. Thresholds are configurable; the labels are LOW / MODERATE / HIGH.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

from smartcity_vision.detection.detector import DetectionResult
from smartcity_vision.utils.config import DensityConfig
from smartcity_vision.utils.geometry import point_in_polygon

CongestionLevel = Literal["LOW", "MODERATE", "HIGH"]

_VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})


@dataclass(frozen=True, slots=True)
class DensityReading:
    """Density snapshot for one frame.

    Attributes:
        vehicles_in_region: Tracked vehicles whose centre is inside the region
            (or in the whole frame when no region is configured).
        vehicles_in_frame: Tracked vehicles anywhere in the frame.
        occupancy_ratio: ``vehicles_in_region / max_expected``, capped at 1.0.
            This is an approximation, not a measured road-area occupancy.
        rolling_average: Mean vehicles-in-region over the configured window.
        congestion: LOW / MODERATE / HIGH from the rolling average.
    """

    vehicles_in_region: int
    vehicles_in_frame: int
    occupancy_ratio: float
    rolling_average: float
    congestion: CongestionLevel

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable reading."""
        return {
            "vehicles_in_region": self.vehicles_in_region,
            "vehicles_in_frame": self.vehicles_in_frame,
            "occupancy_ratio": round(self.occupancy_ratio, 3),
            "rolling_average": round(self.rolling_average, 2),
            "congestion": self.congestion,
        }


def classify_congestion(average: float, moderate: float, high: float) -> CongestionLevel:
    """Map a rolling vehicle count onto a congestion label.

    Thresholds are exclusive of the lower bound: ``average < moderate`` is LOW,
    ``moderate <= average < high`` is MODERATE, otherwise HIGH.
    """
    if average >= high:
        return "HIGH"
    if average >= moderate:
        return "MODERATE"
    return "LOW"


class DensityEstimator:
    """Rolling vehicle-count density with a congestion label."""

    def __init__(
        self, config: DensityConfig, region: tuple[tuple[float, float], ...] | None
    ) -> None:
        """Initialise the estimator.

        Args:
            config: Validated density configuration.
            region: Optional polygon. ``None`` uses the whole frame.
        """
        self._config = config
        self._region = region
        self._window: deque[int] = deque(maxlen=config.window_frames)
        self._history: list[DensityReading] = []
        self._current: DensityReading | None = None

    def update(self, result: DetectionResult) -> DensityReading:
        """Fold one frame into the rolling window.

        Args:
            result: Tracked detections for one frame.

        Returns:
            The density reading for this frame.
        """
        vehicles = [
            detection for detection in result.tracked if detection.class_name in _VEHICLE_CLASSES
        ]
        in_frame = len(vehicles)
        if self._region is None:
            in_region = in_frame
        else:
            in_region = sum(1 for item in vehicles if point_in_polygon(item.center, self._region))

        self._window.append(in_region)
        average = sum(self._window) / len(self._window)
        occupancy = min(1.0, in_region / self._config.max_expected_vehicles)
        reading = DensityReading(
            vehicles_in_region=in_region,
            vehicles_in_frame=in_frame,
            occupancy_ratio=occupancy,
            rolling_average=average,
            congestion=classify_congestion(
                average, self._config.moderate_threshold, self._config.high_threshold
            ),
        )
        self._current = reading
        self._history.append(reading)
        return reading

    @property
    def current(self) -> DensityReading | None:
        """Most recent reading, or ``None`` before the first frame."""
        return self._current

    @property
    def history(self) -> tuple[DensityReading, ...]:
        """Every reading recorded this run, oldest first."""
        return tuple(self._history)

    def peak_congestion(self) -> CongestionLevel:
        """Highest congestion label observed this run."""
        rank = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
        if not self._history:
            return "LOW"
        return max(self._history, key=lambda reading: rank[reading.congestion]).congestion

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the run."""
        current = self._current
        return {
            "current": None if current is None else current.as_dict(),
            "peak_congestion": self.peak_congestion(),
            "samples": len(self._history),
        }


__all__ = ["CongestionLevel", "DensityEstimator", "DensityReading", "classify_congestion"]
