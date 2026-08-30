"""Behaviour of polygonal zone enter / exit / dwell."""

from __future__ import annotations

import pytest

from smartcity_vision.analytics.zones import ZoneMonitor
from smartcity_vision.detection.detector import Detection, DetectionResult
from smartcity_vision.utils.config import ZoneConfig


def zone() -> ZoneConfig:
    return ZoneConfig(
        name="lot",
        kind="road_segment",
        polygon=((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)),
    )


def result(
    frame_index: int,
    timestamp: float,
    track_id: int | None,
    center: tuple[float, float],
    class_name: str = "car",
) -> DetectionResult:
    x, y = center
    return DetectionResult(
        frame_index=frame_index,
        timestamp=timestamp,
        detections=(
            Detection(
                class_id=2,
                class_name=class_name,
                confidence=0.9,
                bbox=(x - 2, y - 2, x + 2, y + 2),
                track_id=track_id,
            ),
        ),
        inference_ms=1.0,
    )


def test_entering_the_polygon_fires_enter_and_occupancy() -> None:
    monitor = ZoneMonitor((zone(),))

    events = monitor.update(result(0, 0.0, 4, (50.0, 50.0)))

    assert [event.kind for event in events] == ["enter"]
    assert events[0].track_id == 4
    assert events[0].dwell_seconds == 0.0
    assert monitor.occupants_in("lot") == 1


def test_staying_inside_does_not_reenter() -> None:
    monitor = ZoneMonitor((zone(),))

    monitor.update(result(0, 0.0, 4, (50.0, 50.0)))
    events = monitor.update(result(1, 0.1, 4, (60.0, 60.0)))

    assert events == ()
    assert monitor.occupants_in("lot") == 1


def test_leaving_the_polygon_fires_exit_with_measured_dwell() -> None:
    monitor = ZoneMonitor((zone(),))

    monitor.update(result(0, 1.0, 4, (50.0, 50.0)))
    events = monitor.update(result(5, 3.5, 4, (150.0, 50.0)))

    assert len(events) == 1
    assert events[0].kind == "exit"
    assert events[0].dwell_seconds == pytest.approx(2.5)
    assert monitor.occupants_in("lot") == 0


def test_a_track_that_vanishes_inside_is_closed_as_an_exit() -> None:
    monitor = ZoneMonitor((zone(),))

    monitor.update(result(0, 0.0, 8, (50.0, 50.0)))
    events = monitor.update(
        DetectionResult(frame_index=10, timestamp=4.0, detections=(), inference_ms=1.0)
    )

    assert [event.kind for event in events] == ["exit"]
    assert events[0].dwell_seconds == pytest.approx(4.0)
    assert events[0].class_name == "unknown"


def test_a_point_on_the_boundary_counts_as_inside() -> None:
    monitor = ZoneMonitor((zone(),))

    events = monitor.update(result(0, 0.0, 1, (0.0, 50.0)))

    assert [event.kind for event in events] == ["enter"]


def test_two_tracks_occupy_independently() -> None:
    monitor = ZoneMonitor((zone(),))
    first = result(0, 0.0, 1, (20.0, 20.0), class_name="car")
    second = DetectionResult(
        frame_index=0,
        timestamp=0.0,
        detections=first.detections
        + (
            Detection(
                class_id=0,
                class_name="person",
                confidence=0.8,
                bbox=(70.0, 70.0, 74.0, 74.0),
                track_id=2,
            ),
        ),
        inference_ms=1.0,
    )

    monitor.update(second)

    occupancy = monitor.occupancy_of("lot")
    assert occupancy is not None
    assert occupancy.occupants == 2
    assert occupancy.by_class == {"car": 1, "person": 1}


def test_untracked_detections_do_not_occupy() -> None:
    monitor = ZoneMonitor((zone(),))

    monitor.update(result(0, 0.0, None, (50.0, 50.0)))

    assert monitor.occupants_in("lot") == 0
    assert monitor.summary().enters == 0
