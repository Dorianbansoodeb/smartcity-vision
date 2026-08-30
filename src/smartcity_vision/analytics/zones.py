"""Polygonal zone enter / exit / dwell detection.

A zone is a closed polygon from config. Membership is decided by the object's
centre, not its bounding box: a box that merely overlaps a zone edge does not
count as inside. Enter and exit fire on the transition; dwell is the
accumulated time the track spent inside, using the frame timestamps so a
variable frame rate does not invent duration.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from smartcity_vision.detection.detector import DetectionResult
from smartcity_vision.utils.config import ZoneConfig
from smartcity_vision.utils.geometry import Point, point_in_polygon
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)

ZoneEventKind = Literal["enter", "exit"]


@dataclass(frozen=True, slots=True)
class ZoneEvent:
    """One enter or exit of a configured zone.

    Attributes:
        zone_name: Configured name of the zone.
        zone_kind: Semantic kind from config (intersection, restricted, ...).
        track_id: Identity of the object.
        class_name: Class at the moment of the event.
        kind: ``"enter"`` or ``"exit"``.
        frame_index: Frame on which the transition was observed.
        timestamp: Seconds since the start of the stream.
        dwell_seconds: For an exit, time spent inside; ``0.0`` on enter.
        position: Object centre at the event frame.
    """

    zone_name: str
    zone_kind: str
    track_id: int
    class_name: str
    kind: ZoneEventKind
    frame_index: int
    timestamp: float
    dwell_seconds: float
    position: Point

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of this event."""
        return {
            "zone_name": self.zone_name,
            "zone_kind": self.zone_kind,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "kind": self.kind,
            "frame_index": self.frame_index,
            "timestamp": round(self.timestamp, 3),
            "dwell_seconds": round(self.dwell_seconds, 3),
            "position": [round(self.position[0], 2), round(self.position[1], 2)],
        }


@dataclass(frozen=True, slots=True)
class ZoneOccupancy:
    """Who is inside a zone on the current frame."""

    zone_name: str
    occupants: int
    by_class: dict[str, int]


@dataclass(frozen=True, slots=True)
class ZoneSummary:
    """Aggregated zone events and current occupancy."""

    events: tuple[ZoneEvent, ...]
    occupancy: dict[str, ZoneOccupancy]
    enters: int
    exits: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "enters": self.enters,
            "exits": self.exits,
            "occupancy": {
                name: {"occupants": occ.occupants, "by_class": dict(sorted(occ.by_class.items()))}
                for name, occ in sorted(self.occupancy.items())
            },
            "events": [event.as_dict() for event in self.events],
        }


class ZoneMonitor:
    """Tracks enter, exit, and dwell for configured polygonal zones."""

    def __init__(self, zones: tuple[ZoneConfig, ...]) -> None:
        """Initialise the monitor.

        Args:
            zones: Validated zone configurations. An empty tuple is valid.
        """
        self._zones = zones
        self._inside: dict[tuple[str, int], float] = {}
        self._events: list[ZoneEvent] = []
        self._current_occupancy: dict[str, ZoneOccupancy] = {
            zone.name: ZoneOccupancy(zone_name=zone.name, occupants=0, by_class={})
            for zone in zones
        }

    def update(self, result: DetectionResult) -> tuple[ZoneEvent, ...]:
        """Inspect one frame and return any enter/exit events it produced.

        A track that vanishes while inside is treated as an implicit exit so
        dwell is closed rather than left hanging until the next time that ID
        is reused.

        Args:
            result: Tracked detections for one frame.

        Returns:
            The enter/exit events observed on this frame.
        """
        if not self._zones:
            return ()

        present: dict[int, tuple[str, Point]] = {}
        for detection in result.tracked:
            if detection.track_id is None:
                continue
            present[detection.track_id] = (detection.class_name, detection.center)

        new_events: list[ZoneEvent] = []
        occupancy: dict[str, Counter[str]] = {zone.name: Counter() for zone in self._zones}

        for zone in self._zones:
            for track_id, (class_name, center) in present.items():
                key = (zone.name, track_id)
                is_inside = point_in_polygon(center, zone.polygon)
                was_inside = key in self._inside
                if is_inside:
                    occupancy[zone.name][class_name] += 1
                    if not was_inside:
                        self._inside[key] = result.timestamp
                        new_events.append(
                            self._event(zone, track_id, class_name, "enter", result, center, 0.0)
                        )
                elif was_inside:
                    entered_at = self._inside.pop(key)
                    new_events.append(
                        self._event(
                            zone,
                            track_id,
                            class_name,
                            "exit",
                            result,
                            center,
                            max(0.0, result.timestamp - entered_at),
                        )
                    )

        vanished = [key for key in list(self._inside) if key[1] not in present]
        for zone_name, track_id in vanished:
            entered_at = self._inside.pop((zone_name, track_id))
            zone = next(item for item in self._zones if item.name == zone_name)
            new_events.append(
                self._event(
                    zone,
                    track_id,
                    "unknown",
                    "exit",
                    result,
                    (0.0, 0.0),
                    max(0.0, result.timestamp - entered_at),
                )
            )

        self._current_occupancy = {
            name: ZoneOccupancy(
                zone_name=name,
                occupants=sum(counts.values()),
                by_class=dict(counts),
            )
            for name, counts in occupancy.items()
        }
        self._events.extend(new_events)
        return tuple(new_events)

    def occupancy_of(self, zone_name: str) -> ZoneOccupancy | None:
        """Return current occupancy for ``zone_name``, or ``None`` if unknown."""
        return self._current_occupancy.get(zone_name)

    def occupants_in(self, zone_name: str) -> int:
        """Number of tracked objects currently inside ``zone_name``."""
        occupancy = self._current_occupancy.get(zone_name)
        return 0 if occupancy is None else occupancy.occupants

    def summary(self) -> ZoneSummary:
        """Return every event recorded so far plus current occupancy."""
        enters = sum(1 for event in self._events if event.kind == "enter")
        exits = sum(1 for event in self._events if event.kind == "exit")
        return ZoneSummary(
            events=tuple(self._events),
            occupancy=dict(self._current_occupancy),
            enters=enters,
            exits=exits,
        )

    @staticmethod
    def _event(
        zone: ZoneConfig,
        track_id: int,
        class_name: str,
        kind: ZoneEventKind,
        result: DetectionResult,
        position: Point,
        dwell_seconds: float,
    ) -> ZoneEvent:
        """Build one event from the current observation."""
        return ZoneEvent(
            zone_name=zone.name,
            zone_kind=zone.kind,
            track_id=track_id,
            class_name=class_name,
            kind=kind,
            frame_index=result.frame_index,
            timestamp=result.timestamp,
            dwell_seconds=dwell_seconds,
            position=position,
        )


__all__ = [
    "ZoneEvent",
    "ZoneMonitor",
    "ZoneOccupancy",
    "ZoneSummary",
]
