"""Behaviour of density estimation and congestion classification."""

from __future__ import annotations

import pytest

from smartcity_vision.analytics.density import DensityEstimator, classify_congestion
from smartcity_vision.detection.detector import Detection, DetectionResult
from smartcity_vision.utils.config import DensityConfig


def result(frame_index: int, *centers: tuple[str, tuple[float, float]]) -> DetectionResult:
    detections = tuple(
        Detection(
            class_id=2,
            class_name=class_name,
            confidence=0.9,
            bbox=(x - 1, y - 1, x + 1, y + 1),
            track_id=index + 1,
        )
        for index, (class_name, (x, y)) in enumerate(centers)
    )
    return DetectionResult(
        frame_index=frame_index,
        timestamp=frame_index / 10.0,
        detections=detections,
        inference_ms=1.0,
    )


@pytest.mark.parametrize(
    ("average", "expected"),
    [
        (0.0, "LOW"),
        (2.9, "LOW"),
        (3.0, "MODERATE"),
        (5.9, "MODERATE"),
        (6.0, "HIGH"),
        (20.0, "HIGH"),
    ],
)
def test_congestion_thresholds_are_inclusive_of_the_lower_bound(
    average: float, expected: str
) -> None:
    assert classify_congestion(average, moderate=3.0, high=6.0) == expected


def test_pedestrians_do_not_count_as_density() -> None:
    estimator = DensityEstimator(
        DensityConfig(window_frames=5, moderate_threshold=3, high_threshold=6),
        region=None,
    )

    reading = estimator.update(result(0, ("person", (10.0, 10.0)), ("car", (20.0, 20.0))))

    assert reading.vehicles_in_frame == 1
    assert reading.congestion == "LOW"


def test_a_region_excludes_vehicles_outside_the_polygon() -> None:
    estimator = DensityEstimator(
        DensityConfig(window_frames=5, moderate_threshold=3, high_threshold=6),
        region=((0.0, 0.0), (50.0, 0.0), (50.0, 50.0), (0.0, 50.0)),
    )

    reading = estimator.update(
        result(0, ("car", (10.0, 10.0)), ("car", (80.0, 80.0)), ("truck", (20.0, 20.0)))
    )

    assert reading.vehicles_in_frame == 3
    assert reading.vehicles_in_region == 2


def test_rolling_average_prevents_a_single_spike_from_flipping_high() -> None:
    estimator = DensityEstimator(
        DensityConfig(window_frames=4, moderate_threshold=3, high_threshold=6),
        region=None,
    )

    for index in range(3):
        estimator.update(result(index, ("car", (1.0, 1.0))))
    spike = estimator.update(
        result(
            3,
            ("car", (1.0, 1.0)),
            ("car", (2.0, 2.0)),
            ("car", (3.0, 3.0)),
            ("car", (4.0, 4.0)),
            ("car", (5.0, 5.0)),
            ("car", (6.0, 6.0)),
            ("car", (7.0, 7.0)),
        )
    )

    assert spike.vehicles_in_region == 7
    assert spike.rolling_average == pytest.approx(2.5)
    assert spike.congestion == "LOW"
    assert estimator.peak_congestion() == "LOW"


def test_occupancy_is_capped_at_one() -> None:
    estimator = DensityEstimator(
        DensityConfig(
            window_frames=1, max_expected_vehicles=2, moderate_threshold=10, high_threshold=20
        ),
        region=None,
    )

    reading = estimator.update(result(0, ("car", (1, 1)), ("car", (2, 2)), ("car", (3, 3))))

    assert reading.occupancy_ratio == pytest.approx(1.0)
