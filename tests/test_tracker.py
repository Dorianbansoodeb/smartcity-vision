"""Behaviour of the tracking wrapper.

The Ultralytics model is stubbed, so these tests assert that we call the
tracking API correctly and convert its identities faithfully, without
downloading weights or running inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from helpers import FakeYolo
from smartcity_vision.detection import detector as detector_module
from smartcity_vision.detection.tracker import YoloTracker
from smartcity_vision.utils.config import ModelConfig, TrackingConfig
from smartcity_vision.video.source import Frame


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeYolo:
    """Install a stub model built up front, so tests can queue results on it."""
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"not-a-real-checkpoint")
    model = FakeYolo(str(weights))
    monkeypatch.setattr(detector_module, "YOLO", lambda _: model)
    return model


def make_tracker(weights: Path, **tracking: Any) -> YoloTracker:
    return YoloTracker(ModelConfig(weights=weights), TrackingConfig(**tracking))


def frame(index: int) -> Frame:
    return Frame(index=index, timestamp=index / 10.0, image=np.zeros((48, 64, 3), dtype=np.uint8))


def test_track_ids_are_attached_to_detections(stub: FakeYolo, tmp_path: Path) -> None:
    stub.queue([(0.0, 0.0, 10.0, 10.0, 0.9, 2)], ids=[4])
    tracker = make_tracker(tmp_path / "yolov8n.pt")

    result = tracker.detect(frame(0))

    assert [item.track_id for item in result.detections] == [4]
    assert result.detections[0].class_name == "car"


def test_ids_persist_across_frames_for_the_same_object(stub: FakeYolo, tmp_path: Path) -> None:
    for _ in range(3):
        stub.queue([(0.0, 0.0, 10.0, 10.0, 0.9, 2)], ids=[11])
    tracker = make_tracker(tmp_path / "yolov8n.pt")

    ids = [tracker.detect(frame(index)).detections[0].track_id for index in range(3)]

    assert ids == [11, 11, 11]


def test_persist_and_tracker_name_are_passed_to_the_model(stub: FakeYolo, tmp_path: Path) -> None:
    stub.queue([], ids=None)
    tracker = make_tracker(tmp_path / "yolov8n.pt", tracker="botsort.yaml")

    tracker.detect(frame(0))

    kwargs = stub.track_kwargs[0]
    # Without persist=True the tracker restarts each call and every object gets a new ID.
    assert kwargs["persist"] is True
    assert kwargs["tracker"] == "botsort.yaml"
    assert tracker.tracker_name == "botsort.yaml"
    # Detection settings must still reach the model through the tracking call.
    assert kwargs["conf"] == pytest.approx(0.25)
    assert sorted(kwargs["classes"]) == [0, 1, 2, 3, 5, 7]


def test_unconfirmed_tracks_yield_detections_without_ids(stub: FakeYolo, tmp_path: Path) -> None:
    # Ultralytics leaves boxes.id as None until the tracker confirms a track.
    stub.queue([(0.0, 0.0, 10.0, 10.0, 0.9, 2)], ids=None)
    tracker = make_tracker(tmp_path / "yolov8n.pt")

    result = tracker.detect(frame(0))

    assert len(result) == 1
    assert result.detections[0].track_id is None
    assert result.tracked == ()


def test_warmup_uses_prediction_so_it_cannot_pollute_tracker_state(
    stub: FakeYolo, tmp_path: Path
) -> None:
    tracker = make_tracker(tmp_path / "yolov8n.pt")

    tracker.warmup(width=64, height=48)

    assert len(stub.predict_kwargs) == 1
    assert stub.track_kwargs == []


def test_a_tracker_is_usable_wherever_a_detector_is(stub: FakeYolo, tmp_path: Path) -> None:
    stub.queue([(0.0, 0.0, 10.0, 10.0, 0.9, 2)], ids=[1])
    tracker = make_tracker(tmp_path / "yolov8n.pt")

    result = tracker.detect(frame(0))

    # Same return type as YoloDetector.detect, so the pipeline is unchanged.
    assert result.frame_index == 0
    assert result.inference_ms > 0.0
    assert hasattr(tracker, "device")
