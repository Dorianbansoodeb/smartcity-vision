"""End-to-end behaviour of the frame-processing pipeline.

A stub detector stands in for YOLO so the pipeline's control flow (frame
skipping, frame limits, video writing, statistics) is tested without inference.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from smartcity_vision.detection.detector import Detection, DetectionResult
from smartcity_vision.exceptions import VideoSourceError
from smartcity_vision.utils.config import AppConfig
from smartcity_vision.video.processor import (
    ProcessingStats,
    VideoProcessor,
    create_source_from_config,
)
from smartcity_vision.video.source import FileVideoSource, Frame


class StubDetector:
    """Returns one car detection per frame, plus a person on even frames."""

    device = "cpu"

    def __init__(self) -> None:
        self.seen_frame_indices: list[int] = []
        self.warmup_calls: list[tuple[int, int]] = []

    def warmup(self, width: int, height: int) -> float:
        self.warmup_calls.append((width, height))
        return 0.0

    def detect(self, frame: Frame) -> DetectionResult:
        self.seen_frame_indices.append(frame.index)
        detections = [
            Detection(class_id=2, class_name="car", confidence=0.9, bbox=(5.0, 5.0, 25.0, 25.0))
        ]
        if frame.index % 2 == 0:
            detections.append(
                Detection(
                    class_id=0,
                    class_name="person",
                    confidence=0.7,
                    bbox=(30.0, 10.0, 40.0, 35.0),
                )
            )
        return DetectionResult(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            detections=tuple(detections),
            inference_ms=1.0 + frame.index,
        )


def make_config(
    source: Path,
    output_dir: Path,
    sections: dict[str, dict[str, object]] | None = None,
) -> AppConfig:
    """Build a config for ``source`` with optional per-section overrides."""
    payload: dict[str, dict[str, object]] = {
        "video": {"source": str(source)},
        "output": {"directory": output_dir},
    }
    for section, values in (sections or {}).items():
        payload[section] = {**payload.get(section, {}), **values}
    return AppConfig.model_validate(payload)


def test_run_processes_every_frame_and_writes_an_annotated_video(
    synthetic_video: Path, tmp_path: Path
) -> None:
    detector = StubDetector()
    config = make_config(synthetic_video, tmp_path / "out")

    stats = VideoProcessor(config, detector).run()  # type: ignore[arg-type]

    assert stats.frames_read == 5
    assert stats.frames_processed == 5
    assert stats.frames_skipped == 0
    assert detector.seen_frame_indices == [0, 1, 2, 3, 4]
    assert stats.total_detections == 8  # 5 cars + 3 persons on even frames
    assert stats.detections_by_class == {"car": 5, "person": 3}
    assert stats.device == "cpu"
    assert stats.interrupted is False
    assert stats.elapsed_seconds > 0

    output = stats.output_video
    assert output is not None and output.is_file() and output.stat().st_size > 0

    capture = cv2.VideoCapture(str(output))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    finally:
        capture.release()


def test_warmup_runs_at_source_resolution_and_is_excluded_from_stats(
    synthetic_video: Path, tmp_path: Path
) -> None:
    detector = StubDetector()
    config = make_config(synthetic_video, tmp_path / "out")

    stats = VideoProcessor(config, detector).run()  # type: ignore[arg-type]

    assert detector.warmup_calls == [(64, 48)]
    # The warmup frame must not be counted as processed work.
    assert stats.frames_processed == 5
    assert len(stats.inference_ms_samples) == 5


def test_warmup_can_be_disabled(synthetic_video: Path, tmp_path: Path) -> None:
    detector = StubDetector()
    config = make_config(synthetic_video, tmp_path / "out", {"model": {"warmup": False}})

    VideoProcessor(config, detector).run()  # type: ignore[arg-type]

    assert detector.warmup_calls == []


def test_frame_skip_processes_a_strided_subset(synthetic_video: Path, tmp_path: Path) -> None:
    detector = StubDetector()
    config = make_config(synthetic_video, tmp_path / "out", {"video": {"frame_skip": 1}})

    stats = VideoProcessor(config, detector).run()  # type: ignore[arg-type]

    assert detector.seen_frame_indices == [0, 2, 4]
    assert stats.frames_read == 5
    assert stats.frames_processed == 3
    assert stats.frames_skipped == 2


def test_max_frames_stops_the_run_early(synthetic_video: Path, tmp_path: Path) -> None:
    detector = StubDetector()
    config = make_config(synthetic_video, tmp_path / "out", {"video": {"max_frames": 2}})

    stats = VideoProcessor(config, detector).run()  # type: ignore[arg-type]

    assert stats.frames_processed == 2
    assert detector.seen_frame_indices == [0, 1]


def test_disabling_video_output_writes_no_file(synthetic_video: Path, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    config = make_config(synthetic_video, output_dir, {"output": {"write_annotated_video": False}})

    stats = VideoProcessor(config, StubDetector()).run()  # type: ignore[arg-type]

    assert stats.frames_processed == 5
    assert stats.output_video is None
    assert not (output_dir / "annotated_video.mp4").exists()


def test_annotation_does_not_mutate_the_source_frame(synthetic_video: Path, tmp_path: Path) -> None:
    config = make_config(synthetic_video, tmp_path / "out")
    processor = VideoProcessor(config, StubDetector())  # type: ignore[arg-type]
    original = np.zeros((48, 64, 3), dtype=np.uint8)
    frame = Frame(index=0, timestamp=0.0, image=original)
    result = StubDetector().detect(frame)

    annotated = processor._annotate(frame, result)  # noqa: SLF001 - internal step under test

    assert annotated is not original
    assert not original.any(), "the decoded frame must be left untouched"
    assert annotated.any(), "the copy must carry the overlay"


def test_source_from_config_dispatches_and_validates_before_use(
    synthetic_video: Path, tmp_path: Path
) -> None:
    config = make_config(synthetic_video, tmp_path / "out")

    source = create_source_from_config(config)

    assert isinstance(source, FileVideoSource)

    missing = make_config(synthetic_video, tmp_path / "out")
    missing = missing.model_copy(
        update={"video": missing.video.model_copy(update={"source": "no/such/clip.mp4"})}
    )
    with pytest.raises(VideoSourceError, match="not found"):
        create_source_from_config(missing)


def test_an_injected_source_is_used_instead_of_the_configured_one(
    synthetic_video: Path, tmp_path: Path
) -> None:
    config = make_config(synthetic_video, tmp_path / "out", {"video": {"source": "0"}})
    detector = StubDetector()

    stats = VideoProcessor(  # type: ignore[arg-type]
        config, detector, source=FileVideoSource(synthetic_video)
    ).run()

    assert stats.frames_processed == 5


def test_stats_summary_reports_measured_latency_percentiles() -> None:
    stats = ProcessingStats(
        frames_processed=4,
        elapsed_seconds=2.0,
        inference_ms_samples=[10.0, 20.0, 30.0, 100.0],
    )

    assert stats.pipeline_fps == pytest.approx(2.0)
    assert stats.avg_inference_ms == pytest.approx(40.0)
    assert stats.p95_inference_ms == pytest.approx(100.0)
    assert stats.as_dict()["avg_inference_ms"] == pytest.approx(40.0)


def test_stats_summary_is_safe_before_any_frame_is_processed() -> None:
    stats = ProcessingStats()

    assert stats.pipeline_fps == 0.0
    assert stats.avg_inference_ms == 0.0
    assert stats.p95_inference_ms == 0.0
