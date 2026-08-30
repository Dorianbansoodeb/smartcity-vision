"""Behaviour of the capped trajectory store."""

from __future__ import annotations

from smartcity_vision.analytics.trajectories import TrajectoryStore
from smartcity_vision.detection.detector import Detection, DetectionResult
from smartcity_vision.utils.config import TrajectoryConfig


def result(frame_index: int, track_id: int, center: tuple[float, float]) -> DetectionResult:
    x, y = center
    return DetectionResult(
        frame_index=frame_index,
        timestamp=frame_index / 10.0,
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


def test_history_is_capped_at_the_configured_length() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=3, forget_after_frames=90))

    for index in range(10):
        store.update(result(index, 5, (float(index), 0.0)))

    trail = store.trail(5)
    assert trail is not None
    assert [sample[0] for sample in trail.points] == [7, 8, 9]
    assert trail.centers == ((7.0, 0.0), (8.0, 0.0), (9.0, 0.0))


def test_a_stale_track_is_forgotten() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=10, forget_after_frames=2))

    store.update(result(0, 1, (0.0, 0.0)))
    store.update(result(1, 1, (1.0, 0.0)))
    store.update(result(5, 2, (0.0, 4.0)))

    assert store.trail(1) is None
    assert store.trail(2) is not None
    assert len(store) == 1


def test_polyline_length_follows_the_centres() -> None:
    store = TrajectoryStore(TrajectoryConfig(history_length=10, forget_after_frames=90))

    store.update(result(0, 1, (0.0, 0.0)))
    store.update(result(1, 1, (3.0, 0.0)))
    store.update(result(2, 1, (3.0, 4.0)))

    trail = store.trail(1)
    assert trail is not None
    assert trail.length_px == 7.0
