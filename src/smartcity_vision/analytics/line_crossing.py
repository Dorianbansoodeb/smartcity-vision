"""Counting-line crossings.

A crossing is defined geometrically, not by proximity: the segment joining a
track's previous centre to its current centre must intersect the counting line,
and the two centres must lie on opposite sides. That combination is what
distinguishes "walked across the line" from "walked past the end of it".

Direction is relative to the directed line ``start→end``: A is the left side
of that vector, B the right side. In image coordinates (y growing downward)
that means A is below a left-to-right line. A track that recrosses the same
line is recorded again;
the same track cannot fire twice on consecutive frames without actually
crossing back.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from smartcity_vision.detection.detector import Detection, DetectionResult
from smartcity_vision.utils.config import CountingLineConfig
from smartcity_vision.utils.geometry import Point, segments_intersect, side_of_line
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)

Direction = str  # "A->B" or "B->A"


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    """One confirmed crossing of a counting line.

    Attributes:
        line_name: Configured name of the line that was crossed.
        track_id: Identity of the object that crossed.
        class_name: Class at the moment of the crossing.
        direction: ``"A->B"`` (left to right of the directed line) or ``"B->A"``.
        frame_index: Frame on which the crossing was observed.
        timestamp: Seconds since the start of the stream.
        position: Object centre at the crossing frame.
    """

    line_name: str
    track_id: int
    class_name: str
    direction: Direction
    frame_index: int
    timestamp: float
    position: Point

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this event."""
        return {
            "line_name": self.line_name,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "direction": self.direction,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "position": [round(self.position[0], 2), round(self.position[1], 2)],
        }


@dataclass(frozen=True, slots=True)
class CrossingSummary:
    """Aggregated crossings at a point in a run."""

    events: tuple[CrossingEvent, ...]
    counts_by_line: dict[str, int]
    counts_by_direction: dict[str, int]
    counts_by_class: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "total": len(self.events),
            "counts_by_line": dict(sorted(self.counts_by_line.items())),
            "counts_by_direction": dict(sorted(self.counts_by_direction.items())),
            "counts_by_class": dict(sorted(self.counts_by_class.items())),
            "events": [event.as_dict() for event in self.events],
        }


class LineCrossingDetector:
    """Detects directed crossings of configured counting lines."""

    def __init__(self, lines: tuple[CountingLineConfig, ...]) -> None:
        """Initialise the detector.

        Args:
            lines: Validated counting-line configurations. An empty tuple is
                valid and produces no events.
        """
        self._lines = lines
        self._last_center: dict[int, Point] = {}
        self._events: list[CrossingEvent] = []

    def update(self, result: DetectionResult) -> tuple[CrossingEvent, ...]:
        """Inspect one frame and return any crossings it produced.

        Detections without a track ID are ignored: without identity there is no
        previous centre to compare against.

        Args:
            result: Tracked detections for one frame.

        Returns:
            The crossings observed on this frame, in line-then-track order.
        """
        if not self._lines:
            return ()

        new_events: list[CrossingEvent] = []
        seen: set[int] = set()
        for detection in result.tracked:
            if detection.track_id is None:
                continue
            seen.add(detection.track_id)
            previous = self._last_center.get(detection.track_id)
            current = detection.center
            if previous is not None:
                new_events.extend(self._crossings_for(detection, previous, current, result))
            self._last_center[detection.track_id] = current

        stale = [track_id for track_id in self._last_center if track_id not in seen]
        for track_id in stale:
            del self._last_center[track_id]

        self._events.extend(new_events)
        return tuple(new_events)

    def summary(self) -> CrossingSummary:
        """Return every crossing recorded so far."""
        by_line: Counter[str] = Counter()
        by_direction: Counter[str] = Counter()
        by_class: Counter[str] = Counter()
        for event in self._events:
            by_line[event.line_name] += 1
            by_direction[event.direction] += 1
            by_class[event.class_name] += 1
        return CrossingSummary(
            events=tuple(self._events),
            counts_by_line=dict(by_line),
            counts_by_direction=dict(by_direction),
            counts_by_class=dict(by_class),
        )

    def _crossings_for(
        self,
        detection: Detection,
        previous: Point,
        current: Point,
        result: DetectionResult,
    ) -> list[CrossingEvent]:
        """Return crossings of every matching line for this track step."""
        events: list[CrossingEvent] = []
        for line in self._lines:
            if line.classes and detection.class_name not in line.classes:
                continue
            start, end = line.start, line.end
            if not segments_intersect((previous, current), (start, end)):
                continue
            before = side_of_line(previous, start, end)
            after = side_of_line(current, start, end)
            if before == 0 or after == 0 or before == after:
                continue
            direction: Direction = "A->B" if before > 0 else "B->A"
            events.append(
                CrossingEvent(
                    line_name=line.name,
                    track_id=detection.track_id or 0,
                    class_name=detection.class_name,
                    direction=direction,
                    frame_index=result.frame_index,
                    timestamp=result.timestamp,
                    position=current,
                )
            )
        return events


__all__ = [
    "CrossingEvent",
    "CrossingSummary",
    "LineCrossingDetector",
]
