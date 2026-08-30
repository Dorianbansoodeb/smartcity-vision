"""Behaviour of speed and queue estimation."""

from __future__ import annotations

import pytest

from smartcity_vision.analytics.queue import QueueEstimator
from smartcity_vision.analytics.speed import SpeedEstimator
from smartcity_vision.analytics.trajectories import TrajectoryStore
from smartcity_vision.detection.detector import Detection, DetectionResult
from smartcity_vision.utils.config import QueueConfig, SpeedConfig, TrajectoryConfig


def result(
    frame_index: int, timestamp: float, track_id: int, center: tuple[float, float]
) -> DetectionResult:
    x, y = center
    return DetectionResult(
        frame_index=frame_index,
        timestamp=timestamp,
        detections=(
            Detection(
                class_id=2,
                class_name="car",
                confidence=0.9,
                bbox=(x - 1, y - 1, x + 1, y + 1),
                track_id=track_id,
            ),
        ),
        inference_ms=1.0,
    )


def feed(
    store: TrajectoryStore,
    speed: SpeedEstimator,
    frames: list[tuple[int, float, tuple[float, float]]],
    track_id: int = 1,
) -> list:
    readings = []
    for frame_index, timestamp, center in frames:
        item = result(frame_index, timestamp, track_id, center)
        store.update(item)
        readings.append(speed.update(item, store))
    return readings


def test_uncalibrated_speed_is_pixels_per_second() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=10, forget_after_frames=90))
    estimator = SpeedEstimator(SpeedConfig(window_frames=2))

    feed(store, estimator, [(0, 0.0, (0.0, 0.0)), (1, 1.0, (10.0, 0.0))])

    reading = estimator.current[0]
    assert reading.speed_px_s == pytest.approx(10.0)
    assert reading.speed_kmh is None
    assert reading.calibrated is False


def test_reference_points_convert_to_kmh() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=10, forget_after_frames=90))
    # 10 pixels = 1 metre, so 10 px/s = 1 m/s = 3.6 km/h.
    estimator = SpeedEstimator(
        SpeedConfig(
            window_frames=2,
            reference_points=((0.0, 0.0), (10.0, 0.0)),
            reference_distance_m=1.0,
        )
    )

    feed(store, estimator, [(0, 0.0, (0.0, 0.0)), (1, 1.0, (10.0, 0.0))])

    reading = estimator.current[0]
    assert reading.calibrated is True
    assert reading.speed_kmh == pytest.approx(3.6)


def test_metres_per_pixel_is_an_explicit_scale() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=10, forget_after_frames=90))
    estimator = SpeedEstimator(SpeedConfig(window_frames=2, metres_per_pixel=0.1))

    feed(store, estimator, [(0, 0.0, (0.0, 0.0)), (1, 1.0, (10.0, 0.0))])

    assert estimator.current[0].speed_kmh == pytest.approx(3.6)


def test_speed_is_smoothed_across_the_window() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=10, forget_after_frames=90))
    estimator = SpeedEstimator(SpeedConfig(window_frames=2))

    feed(
        store,
        estimator,
        [(0, 0.0, (0.0, 0.0)), (1, 1.0, (10.0, 0.0)), (2, 2.0, (12.0, 0.0))],
    )

    assert estimator.current[0].speed_px_s == pytest.approx(6.0)


def test_a_stopped_vehicle_is_queued_and_a_moving_one_is_not() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=10, forget_after_frames=90))
    queue = QueueEstimator(QueueConfig(window_frames=5, stopped_px_per_sec=5.0))

    stopped = result(0, 0.0, 1, (0.0, 0.0))
    store.update(stopped)
    queue.update(stopped, store)

    stopped = result(1, 1.0, 1, (1.0, 0.0))  # 1 px/s
    store.update(stopped)
    reading = queue.update(stopped, store)

    assert reading.queued_vehicles == 1
    assert reading.calibrated is False
    assert reading.length_m is None


def test_queue_length_converts_when_calibrated() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=10, forget_after_frames=90))
    queue = QueueEstimator(
        QueueConfig(window_frames=5, stopped_px_per_sec=50.0, metres_per_pixel=0.5)
    )

    first = result(0, 0.0, 1, (0.0, 0.0))
    store.update(first)
    second = DetectionResult(
        frame_index=1,
        timestamp=1.0,
        detections=(
            Detection(
                class_id=2, class_name="car", confidence=0.9, bbox=(-1, -1, 1, 1), track_id=1
            ),
            Detection(
                class_id=2, class_name="car", confidence=0.9, bbox=(9, -1, 11, 1), track_id=2
            ),
        ),
        inference_ms=1.0,
    )
    store.update(second)
    # Give track 2 a previous point so it can be classified.
    store.update(
        DetectionResult(
            frame_index=2,
            timestamp=2.0,
            detections=second.detections,
            inference_ms=1.0,
        )
    )
    reading = queue.update(
        DetectionResult(
            frame_index=2, timestamp=2.0, detections=second.detections, inference_ms=1.0
        ),
        store,
    )

    assert reading.queued_vehicles == 2
    assert reading.calibrated is True
    assert reading.length_m == pytest.approx(reading.length_px * 0.5)
