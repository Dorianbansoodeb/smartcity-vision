"""Behaviour of unique object counting.

These tests are the guard against the classic failure of this kind of system:
counting the same vehicle once per frame and reporting a wildly inflated total.
"""

from __future__ import annotations

import pytest

from smartcity_vision.analytics.counter import UniqueObjectCounter
from smartcity_vision.detection.detector import Detection, DetectionResult
from smartcity_vision.utils.config import CountingConfig


def result(frame_index: int, *objects: tuple[int | None, str]) -> DetectionResult:
    """Build a frame result from ``(track_id, class_name)`` pairs."""
    detections = tuple(
        Detection(
            class_id=0,
            class_name=class_name,
            confidence=0.9,
            bbox=(0.0, 0.0, 10.0, 10.0),
            track_id=track_id,
        )
        for track_id, class_name in objects
    )
    return DetectionResult(
        frame_index=frame_index,
        timestamp=frame_index / 10.0,
        detections=detections,
        inference_ms=1.0,
    )


def counter(min_track_frames: int = 1, forget_track_after_frames: int = 90) -> UniqueObjectCounter:
    return UniqueObjectCounter(
        CountingConfig(
            min_track_frames=min_track_frames,
            forget_track_after_frames=forget_track_after_frames,
        )
    )


def test_a_track_present_in_many_frames_counts_once() -> None:
    subject = counter()

    for frame_index in range(50):
        subject.update(result(frame_index, (7, "car")))

    assert subject.total == 1
    assert subject.counts_by_class == {"car": 1}


def test_distinct_tracks_count_separately() -> None:
    subject = counter()

    subject.update(result(0, (1, "car"), (2, "car"), (3, "person")))

    assert subject.total == 3
    assert subject.counts_by_class == {"car": 2, "person": 1}


def test_a_track_that_disappears_and_returns_is_not_recounted() -> None:
    subject = counter(min_track_frames=1, forget_track_after_frames=5)

    subject.update(result(0, (4, "bus")))
    for frame_index in range(1, 40):  # long enough to evict the track's tallies
        subject.update(result(frame_index))
    subject.update(result(40, (4, "bus")))

    assert subject.total == 1
    assert subject.counts_by_class == {"bus": 1}


def test_a_track_below_the_confirmation_threshold_is_not_counted_yet() -> None:
    subject = counter(min_track_frames=3)

    subject.update(result(0, (9, "car")))
    subject.update(result(1, (9, "car")))

    assert subject.total == 0
    assert subject.counts_by_class == {}
    assert subject.pending_tracks == 1

    subject.update(result(2, (9, "car")))

    assert subject.total == 1
    assert subject.pending_tracks == 0


def test_a_one_frame_false_positive_never_counts() -> None:
    subject = counter(min_track_frames=3)

    subject.update(result(0, (1, "car"), (99, "truck")))
    for frame_index in range(1, 5):
        subject.update(result(frame_index, (1, "car")))

    assert subject.counts_by_class == {"car": 1}


def test_class_flicker_resolves_by_majority_without_double_counting() -> None:
    subject = counter(min_track_frames=1)

    # A car briefly misread as a bus, which yolov8n does regularly.
    for frame_index, class_name in enumerate(["car", "bus", "car", "car"]):
        subject.update(result(frame_index, (5, class_name)))

    assert subject.total == 1
    assert subject.counts_by_class == {"car": 1}


def test_a_sustained_class_change_moves_the_count_rather_than_adding_one() -> None:
    subject = counter(min_track_frames=1)

    subject.update(result(0, (5, "car")))
    assert subject.counts_by_class == {"car": 1}

    for frame_index in range(1, 5):
        subject.update(result(frame_index, (5, "truck")))

    assert subject.total == 1
    assert subject.counts_by_class == {"truck": 1}


def test_totals_always_equal_the_sum_of_per_class_counts() -> None:
    subject = counter(min_track_frames=2)

    subject.update(result(0, (1, "car"), (2, "bus"), (3, "person")))
    subject.update(result(1, (1, "truck"), (2, "bus"), (3, "person")))
    subject.update(result(2, (1, "truck"), (4, "bicycle")))

    assert sum(subject.counts_by_class.values()) == subject.total


def test_detections_without_track_ids_are_ignored() -> None:
    subject = counter()

    subject.update(result(0, (None, "car"), (None, "car")))

    assert subject.total == 0
    assert subject.counts_by_class == {}


def test_untracked_detections_are_warned_about_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    subject = counter()

    with caplog.at_level("WARNING"):
        for frame_index in range(5):
            subject.update(result(frame_index, (None, "car")))

    warnings = [record for record in caplog.records if "track IDs" in record.message]
    assert len(warnings) == 1, "the warning must not repeat every frame"


def test_short_lived_tracks_are_reported_as_discarded() -> None:
    subject = counter(min_track_frames=3, forget_track_after_frames=2)

    subject.update(result(0, (1, "car"), (2, "car")))  # 2 flickers into existence
    subject.update(result(1, (1, "car")))
    subject.update(result(2, (1, "car")))  # track 1 confirms here
    for frame_index in range(3, 10):
        subject.update(result(frame_index, (1, "car")))

    assert subject.total == 1
    assert subject.discarded_tracks == 1
    assert subject.summary().discarded_tracks == 1


def test_active_tracks_reflects_only_the_latest_frame() -> None:
    subject = counter()

    subject.update(result(0, (1, "car"), (2, "car")))
    assert subject.active_tracks == 2

    subject.update(result(1, (1, "car")))
    assert subject.active_tracks == 1

    subject.update(result(2))
    assert subject.active_tracks == 0


def test_eviction_bounds_per_track_state_but_keeps_totals() -> None:
    subject = counter(min_track_frames=1, forget_track_after_frames=3)

    for track_id in range(200):
        subject.update(result(track_id, (track_id, "car")))

    assert subject.total == 200
    assert subject.counts_by_class == {"car": 200}
    # Vote tallies for long-gone tracks must not accumulate for the whole run.
    assert len(subject._class_votes) <= 5  # noqa: SLF001 - bounding memory is the point


def test_summary_snapshot_is_json_serialisable() -> None:
    import json

    subject = counter(min_track_frames=2)
    subject.update(result(0, (1, "car"), (2, "person")))
    subject.update(result(1, (1, "car")))

    payload = subject.summary().as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["total"] == 1
    assert payload["pending_tracks"] == 1
    assert payload["tracks_observed"] == 2


def test_summary_is_a_snapshot_not_a_live_view() -> None:
    subject = counter()

    subject.update(result(0, (1, "car")))
    snapshot = subject.summary()
    subject.update(result(1, (2, "car")))

    assert snapshot.total == 1
    assert subject.total == 2
