"""Behaviour of directed counting-line crossings."""

from __future__ import annotations

from smartcity_vision.analytics.line_crossing import LineCrossingDetector
from smartcity_vision.detection.detector import Detection, DetectionResult
from smartcity_vision.utils.config import CountingLineConfig


def line(
    name: str = "mid",
    start: tuple[float, float] = (0.0, 50.0),
    end: tuple[float, float] = (100.0, 50.0),
    classes: tuple[str, ...] = (),
) -> CountingLineConfig:
    return CountingLineConfig(name=name, start=start, end=end, classes=classes)


def result(
    frame_index: int,
    track_id: int | None,
    center: tuple[float, float],
    class_name: str = "car",
) -> DetectionResult:
    x, y = center
    return DetectionResult(
        frame_index=frame_index,
        timestamp=frame_index / 10.0,
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


def test_crossing_left_of_the_directed_line_is_a_to_b() -> None:
    # Image y grows downward, so the left of a left-to-right line is below it.
    detector = LineCrossingDetector((line(),))

    detector.update(result(0, 7, (50.0, 90.0)))
    events = detector.update(result(1, 7, (50.0, 10.0)))

    assert len(events) == 1
    assert events[0].direction == "A->B"
    assert events[0].track_id == 7
    assert events[0].line_name == "mid"
    assert events[0].class_name == "car"


def test_crossing_right_of_the_directed_line_is_b_to_a() -> None:
    detector = LineCrossingDetector((line(),))

    detector.update(result(0, 3, (50.0, 10.0)))
    events = detector.update(result(1, 3, (50.0, 90.0)))

    assert [event.direction for event in events] == ["B->A"]


def test_walking_past_the_end_of_the_line_is_not_a_crossing() -> None:
    detector = LineCrossingDetector((line(),))

    detector.update(result(0, 1, (200.0, 10.0)))
    events = detector.update(result(1, 1, (200.0, 90.0)))

    assert events == ()


def test_a_track_that_never_crosses_produces_no_event() -> None:
    detector = LineCrossingDetector((line(),))

    detector.update(result(0, 1, (20.0, 10.0)))
    events = detector.update(result(1, 1, (80.0, 10.0)))

    assert events == ()


def test_the_same_track_can_recross_in_the_opposite_direction() -> None:
    detector = LineCrossingDetector((line(),))

    detector.update(result(0, 4, (50.0, 90.0)))
    first = detector.update(result(1, 4, (50.0, 10.0)))
    second = detector.update(result(2, 4, (50.0, 90.0)))

    assert [event.direction for event in first + second] == ["A->B", "B->A"]
    assert detector.summary().counts_by_direction == {"A->B": 1, "B->A": 1}


def test_class_filter_ignores_other_classes() -> None:
    detector = LineCrossingDetector((line(classes=("car",)),))

    detector.update(result(0, 1, (50.0, 10.0), class_name="person"))
    events = detector.update(result(1, 1, (50.0, 90.0), class_name="person"))

    assert events == ()


def test_untracked_detections_cannot_cross() -> None:
    detector = LineCrossingDetector((line(),))

    detector.update(result(0, None, (50.0, 10.0)))
    events = detector.update(result(1, None, (50.0, 90.0)))

    assert events == ()


def test_two_lines_can_fire_on_the_same_step() -> None:
    detector = LineCrossingDetector(
        (
            line(name="horizontal", start=(0.0, 50.0), end=(100.0, 50.0)),
            line(name="vertical", start=(50.0, 0.0), end=(50.0, 100.0)),
        )
    )

    detector.update(result(0, 9, (20.0, 20.0)))
    events = detector.update(result(1, 9, (80.0, 80.0)))

    assert {event.line_name for event in events} == {"horizontal", "vertical"}
