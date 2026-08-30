"""Unique object counting.

Counting detections per frame would report the same car hundreds of times, so
counting is keyed on track identity: a track contributes exactly one to the
totals no matter how many frames it appears in.

Two details make the totals defensible rather than merely plausible:

* A track must be observed ``min_track_frames`` times before it counts, which
  stops one-frame false positives from inflating the numbers.
* A track's class is decided by majority vote across its observations, because
  YOLO regularly flickers between visually similar classes (car/truck/bus) on
  the same object. If the majority changes later, the count is moved rather than
  duplicated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from smartcity_vision.detection.detector import DetectionResult
from smartcity_vision.utils.config import CountingConfig
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CountingSummary:
    """Snapshot of unique counts at a point in a run.

    Attributes:
        counts_by_class: Unique confirmed objects per class name.
        total: Total unique confirmed objects.
        active_tracks: Tracks present in the most recently processed frame.
        tracks_observed: Distinct track IDs seen so far, confirmed or not.
        pending_tracks: Tracks seen but not yet at ``min_track_frames``.
        discarded_tracks: Tracks that disappeared before reaching
            ``min_track_frames`` and so never counted. A useful proxy for how
            much short-lived noise the confirmation threshold is filtering out.
    """

    counts_by_class: dict[str, int]
    total: int
    active_tracks: int
    tracks_observed: int
    pending_tracks: int
    discarded_tracks: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "counts_by_class": dict(sorted(self.counts_by_class.items())),
            "total": self.total,
            "active_tracks": self.active_tracks,
            "tracks_observed": self.tracks_observed,
            "pending_tracks": self.pending_tracks,
            "discarded_tracks": self.discarded_tracks,
        }


class UniqueObjectCounter:
    """Counts unique tracked objects per class without double counting.

    Memory is bounded per track: the per-track vote tallies are evicted once a
    track has been missing for ``forget_track_after_frames``. The set of track
    identities and their attributed class is retained for the whole run, because
    that is precisely what guarantees a reappearing ID is not counted twice.
    """

    def __init__(self, config: CountingConfig) -> None:
        """Initialise the counter.

        Args:
            config: Validated counting configuration.
        """
        self._config = config
        self._class_votes: dict[int, Counter[str]] = {}
        self._last_seen_frame: dict[int, int] = {}
        self._counted_class: dict[int, str] = {}
        self._ever_seen: set[int] = set()
        self._counts: Counter[str] = Counter()
        self._active_tracks = 0
        self._discarded_tracks = 0
        self._warned_about_untracked = False

    def update(self, result: DetectionResult) -> None:
        """Fold one frame's detections into the running totals.

        Detections without a track ID are ignored: without identity there is no
        way to tell a new object from one already counted. A single warning is
        logged the first time this happens, since it usually means tracking is
        disabled by mistake.

        Args:
            result: Detections for one frame, normally from a tracker.
        """
        tracked = result.tracked
        if len(tracked) < len(result) and not self._warned_about_untracked:
            self._warned_about_untracked = True
            logger.warning(
                "Ignoring detections without track IDs for counting; "
                "enable tracking to count unique objects"
            )

        self._active_tracks = len(tracked)
        for detection in tracked:
            if detection.track_id is None:
                continue
            self._observe(detection.track_id, detection.class_name, result.frame_index)

        self._evict_stale_tracks(result.frame_index)

    def summary(self) -> CountingSummary:
        """Return the current counts as an immutable snapshot."""
        return CountingSummary(
            counts_by_class=self.counts_by_class,
            total=self.total,
            active_tracks=self._active_tracks,
            tracks_observed=len(self._ever_seen),
            pending_tracks=self.pending_tracks,
            discarded_tracks=self._discarded_tracks,
        )

    @property
    def counts_by_class(self) -> dict[str, int]:
        """Unique confirmed objects per class, excluding emptied classes."""
        return {name: count for name, count in sorted(self._counts.items()) if count > 0}

    @property
    def total(self) -> int:
        """Total unique confirmed objects."""
        return len(self._counted_class)

    @property
    def active_tracks(self) -> int:
        """Tracks present in the most recently processed frame."""
        return self._active_tracks

    @property
    def pending_tracks(self) -> int:
        """Tracks seen but not yet confirmed by ``min_track_frames``."""
        return sum(1 for track_id in self._class_votes if track_id not in self._counted_class)

    @property
    def discarded_tracks(self) -> int:
        """Tracks that vanished before reaching ``min_track_frames``."""
        return self._discarded_tracks

    def _observe(self, track_id: int, class_name: str, frame_index: int) -> None:
        """Record one sighting of ``track_id`` and update counts if confirmed."""
        self._ever_seen.add(track_id)
        self._last_seen_frame[track_id] = frame_index

        votes = self._class_votes.setdefault(track_id, Counter())
        votes[class_name] += 1

        already_counted = track_id in self._counted_class
        if not already_counted and votes.total() < self._config.min_track_frames:
            return

        majority_class = votes.most_common(1)[0][0]
        if not already_counted:
            self._counted_class[track_id] = majority_class
            self._counts[majority_class] += 1
            return

        previous_class = self._counted_class[track_id]
        if previous_class != majority_class:
            # Move the count instead of adding one: it is the same object.
            self._counts[previous_class] -= 1
            self._counts[majority_class] += 1
            self._counted_class[track_id] = majority_class
            logger.debug(
                "Track %d reclassified %s -> %s by majority vote",
                track_id,
                previous_class,
                majority_class,
            )

    def _evict_stale_tracks(self, frame_index: int) -> None:
        """Drop vote tallies for tracks missing longer than the configured window."""
        cutoff = frame_index - self._config.forget_track_after_frames
        stale = [
            track_id for track_id, last_seen in self._last_seen_frame.items() if last_seen < cutoff
        ]
        for track_id in stale:
            del self._last_seen_frame[track_id]
            self._class_votes.pop(track_id, None)
            if track_id not in self._counted_class:
                self._discarded_tracks += 1


__all__ = ["CountingSummary", "UniqueObjectCounter"]
