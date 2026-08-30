"""Behaviour of class-distribution drift detection."""

from __future__ import annotations

import pytest

from smartcity_vision.monitoring.drift import DriftDetector, class_distribution, total_variation
from smartcity_vision.utils.config import MonitoringConfig


def test_identical_distributions_have_zero_total_variation() -> None:
    dist = {"car": 0.5, "person": 0.5}

    assert total_variation(dist, dist) == pytest.approx(0.0)


def test_disjoint_distributions_have_total_variation_one() -> None:
    assert total_variation({"car": 1.0}, {"person": 1.0}) == pytest.approx(1.0)


def test_an_injected_shift_is_flagged() -> None:
    detector = DriftDetector(
        MonitoringConfig(drift_window_frames=10, drift_threshold=0.4, drift_min_detections=10),
        baseline={"car": 80, "person": 20},
    )

    for _ in range(10):
        report = detector.observe("person")

    assert report is not None
    assert report.drifted is True
    assert report.distance == pytest.approx(0.8)
    assert report.window_distribution["person"] == pytest.approx(1.0)


def test_a_matching_window_is_not_flagged() -> None:
    detector = DriftDetector(
        MonitoringConfig(drift_window_frames=10, drift_threshold=0.4, drift_min_detections=10),
        baseline={"car": 8, "person": 2},
    )

    labels = ["car"] * 8 + ["person"] * 2
    report = None
    for label in labels:
        report = detector.observe(label)

    assert report is not None
    assert report.drifted is False
    assert report.distance == pytest.approx(0.0)


def test_a_cold_start_uses_the_first_window_as_baseline() -> None:
    detector = DriftDetector(
        MonitoringConfig(drift_window_frames=5, drift_threshold=0.2, drift_min_detections=5)
    )

    first = [detector.observe("car") for _ in range(5)]
    # The first filled window seeds the baseline rather than alarming against empty.
    assert first[-1] is None

    later = detector.observe("car")
    assert later is not None
    assert later.drifted is False


def test_class_distribution_normalises() -> None:
    assert class_distribution({"car": 3, "bus": 1}) == {"car": 0.75, "bus": 0.25}
    assert class_distribution({}) == {}
