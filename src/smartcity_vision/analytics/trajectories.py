"""Per-track centre-point history.

Each tracked object keeps a rolling list of recent centres so the overlay can
draw a trail and later phases (speed, queue) can measure motion. History length
is capped, so a long or live stream cannot grow memory without bound.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from smartcity_vision.detection.detector import DetectionResult
from smartcity_vision.utils.config import TrajectoryConfig
from smartcity_vision.utils.geometry import Point, polyline_length

PointSample = tuple[int, float, Point]  # frame_index, timestamp, center


@dataclass(frozen=True, slots=True)
class TrackTrail:
    """Recent centres for one track, oldest first.

    Attributes:
        track_id: Identity of the object.
        class_name: Most recently observed class.
        points: ``(frame_index, timestamp, center)`` samples, oldest first.
    """

    track_id: int
    class_name: str
    points: tuple[PointSample, ...]

    @property
    def centers(self) -> tuple[Point, ...]:
        """Centres only, oldest first, for drawing."""
        return tuple(sample[2] for sample in self.points)

    @property
    def length_px(self) -> float:
        """Polyline length of the trail in pixels."""
        return polyline_length(self.centers)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this trail."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "points": [
                {
                    "frame_index": frame_index,
                    "timestamp": round(timestamp, 3),
                    "center": [round(center[0], 2), round(center[1], 2)],
                }
                for frame_index, timestamp, center in self.points
            ],
            "length_px": round(self.length_px, 2),
        }


class TrajectoryStore:
    """Bounded per-track centre history."""

    def __init__(self, config: TrajectoryConfig) -> None:
        """Initialise the store.

        Args:
            config: Validated trajectory configuration.
        """
        self._config = config
        self._trails: dict[int, deque[PointSample]] = {}
        self._class_name: dict[int, str] = {}
        self._last_seen: dict[int, int] = {}

    def update(self, result: DetectionResult) -> None:
        """Append this frame's centres and evict stale tracks.

        Args:
            result: Tracked detections for one frame.
        """
        seen: set[int] = set()
        for detection in result.tracked:
            if detection.track_id is None:
                continue
            seen.add(detection.track_id)
            trail = self._trails.setdefault(
                detection.track_id, deque(maxlen=self._config.history_length)
            )
            trail.append((result.frame_index, result.timestamp, detection.center))
            self._class_name[detection.track_id] = detection.class_name
            self._last_seen[detection.track_id] = result.frame_index

        cutoff = result.frame_index - self._config.forget_after_frames
        stale = [
            track_id
            for track_id, last_seen in self._last_seen.items()
            if last_seen < cutoff and track_id not in seen
        ]
        for track_id in stale:
            self._trails.pop(track_id, None)
            self._class_name.pop(track_id, None)
            del self._last_seen[track_id]

    def trail(self, track_id: int) -> TrackTrail | None:
        """Return the trail for ``track_id``, or ``None`` if unknown."""
        samples = self._trails.get(track_id)
        if not samples:
            return None
        return TrackTrail(
            track_id=track_id,
            class_name=self._class_name.get(track_id, "unknown"),
            points=tuple(samples),
        )

    def active_trails(self) -> tuple[TrackTrail, ...]:
        """Return every trail still being retained, sorted by track ID."""
        return tuple(
            trail
            for track_id in sorted(self._trails)
            if (trail := self.trail(track_id)) is not None
        )

    def __len__(self) -> int:
        """Number of tracks currently retained."""
        return len(self._trails)


__all__ = ["TrackTrail", "TrajectoryStore"]
