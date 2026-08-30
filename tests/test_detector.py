"""Behaviour of the YOLO detector wrapper.

The Ultralytics model is replaced by a stub so these tests assert our own
conversion, filtering, and error handling without downloading weights or running
inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from helpers import install_fake_yolo
from smartcity_vision.detection import detector as detector_module
from smartcity_vision.detection.detector import YoloDetector, resolve_device
from smartcity_vision.exceptions import ConfigError, DetectionError
from smartcity_vision.utils.config import ModelConfig
from smartcity_vision.video.source import Frame


@pytest.fixture
def weights_file(tmp_path: Path) -> Path:
    path = tmp_path / "yolov8n.pt"
    path.write_bytes(b"not-a-real-checkpoint")
    return path


def frame(index: int = 3, timestamp: float = 0.5) -> Frame:
    return Frame(
        index=index,
        timestamp=timestamp,
        image=np.zeros((48, 64, 3), dtype=np.uint8),
    )


def test_detections_are_converted_from_model_output(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    rows = [
        (10.0, 20.0, 50.0, 80.0, 0.91, 2),
        (60.0, 30.0, 90.0, 70.0, 0.42, 7),
    ]
    install_fake_yolo(monkeypatch, rows=rows)
    detector = YoloDetector(ModelConfig(weights=weights_file))

    result = detector.detect(frame(index=7, timestamp=1.25))

    assert result.frame_index == 7
    assert result.timestamp == pytest.approx(1.25)
    assert len(result) == 2
    assert result.inference_ms > 0.0

    first, second = result.detections
    assert (first.class_id, first.class_name) == (2, "car")
    assert first.confidence == pytest.approx(0.91, abs=1e-5)
    assert first.bbox == pytest.approx((10.0, 20.0, 50.0, 80.0))
    assert (second.class_id, second.class_name) == (7, "truck")


def test_empty_model_output_yields_no_detections(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    install_fake_yolo(monkeypatch, rows=[])
    detector = YoloDetector(ModelConfig(weights=weights_file))

    result = detector.detect(frame())

    assert len(result) == 0
    assert result.detections == ()


def test_configured_thresholds_and_classes_are_passed_to_the_model(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    built = install_fake_yolo(monkeypatch, rows=[])
    config = ModelConfig(
        weights=weights_file,
        device="cpu",
        confidence_threshold=0.4,
        iou_threshold=0.55,
        image_size=320,
        target_classes=("car", "bus"),
    )
    detector = YoloDetector(config)

    detector.detect(frame())

    kwargs = built[0].predict_kwargs[0]
    assert kwargs["conf"] == pytest.approx(0.4)
    assert kwargs["iou"] == pytest.approx(0.55)
    assert kwargs["imgsz"] == 320
    assert kwargs["device"] == "cpu"
    assert kwargs["verbose"] is False
    # Filtering happens inside the model, using ids resolved from class names.
    assert sorted(kwargs["classes"]) == [2, 5]


def test_weights_are_loaded_once_and_reused_across_frames(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    built = install_fake_yolo(monkeypatch, rows=[])
    detector = YoloDetector(ModelConfig(weights=weights_file))

    for index in range(5):
        detector.detect(frame(index=index))

    assert len(built) == 1, "model must not be reloaded per frame"
    assert len(built[0].predict_kwargs) == 5


def test_unknown_target_class_is_reported_as_a_config_error(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    install_fake_yolo(monkeypatch, rows=[])

    with pytest.raises(ConfigError, match="skateboard"):
        YoloDetector(ModelConfig(weights=weights_file, target_classes=("car", "skateboard")))


def test_inference_failure_is_wrapped_in_a_detection_error(
    monkeypatch: pytest.MonkeyPatch, weights_file: Path
) -> None:
    install_fake_yolo(monkeypatch, rows=[], raises=True)
    detector = YoloDetector(ModelConfig(weights=weights_file))

    with pytest.raises(DetectionError, match="frame 3"):
        detector.detect(frame(index=3))


def test_missing_weights_are_downloaded_to_the_configured_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    built = install_fake_yolo(monkeypatch, rows=[])
    absent = tmp_path / "models" / "yolov8s.pt"
    requested: list[Path] = []

    def fake_download(path: Path) -> str:
        requested.append(Path(path))
        Path(path).write_bytes(b"downloaded")
        return str(path)

    monkeypatch.setattr(detector_module, "attempt_download_asset", fake_download)

    YoloDetector(ModelConfig(weights=absent))

    # The checkpoint must land at the configured path, not in the working directory.
    assert requested == [absent]
    assert absent.is_file()
    assert built[0].weights == str(absent)


def test_download_failure_is_reported_as_a_detection_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_fake_yolo(monkeypatch, rows=[])

    def failing_download(path: Path) -> str:
        raise OSError("network unreachable")

    monkeypatch.setattr(detector_module, "attempt_download_asset", failing_download)

    with pytest.raises(DetectionError, match="yolov8s.pt"):
        YoloDetector(ModelConfig(weights=tmp_path / "models" / "yolov8s.pt"))


@pytest.mark.parametrize(
    ("requested", "cuda", "mps", "expected"),
    [
        ("auto", True, True, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
        ("cpu", True, True, "cpu"),
        ("mps", False, True, "mps"),
        ("cuda", False, True, "cpu"),
        ("mps", False, False, "cpu"),
    ],
)
def test_device_resolution_prefers_available_accelerators(
    monkeypatch: pytest.MonkeyPatch, requested: str, cuda: bool, mps: bool, expected: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)

    assert resolve_device(requested) == expected
