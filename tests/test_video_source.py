"""Behaviour of the video source abstraction and its factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smartcity_vision.exceptions import VideoSourceError
from smartcity_vision.video.source import (
    FileVideoSource,
    StreamVideoSource,
    WebcamVideoSource,
    create_video_source,
)


def test_file_source_yields_frames_in_order_with_metadata(synthetic_video: Path) -> None:
    with FileVideoSource(synthetic_video) as source:
        metadata = source.metadata
        frames = list(source)

    assert metadata.width == 64
    assert metadata.height == 48
    assert metadata.fps == pytest.approx(10.0, abs=0.5)
    assert metadata.frame_count == 5
    assert metadata.is_live is False

    assert [frame.index for frame in frames] == [0, 1, 2, 3, 4]
    assert all(frame.image.shape == (48, 64, 3) for frame in frames)
    # Grey level increases per frame, so decoded content must be monotonic.
    means = [float(np.mean(frame.image)) for frame in frames]
    assert means == sorted(means)
    assert means[-1] > means[0]


def test_file_source_timestamps_follow_frame_rate(synthetic_video: Path) -> None:
    with FileVideoSource(synthetic_video) as source:
        fps = source.metadata.fps
        frames = list(source)

    for frame in frames:
        assert frame.timestamp == pytest.approx(frame.index / fps)


def test_looping_file_source_continues_past_the_end(synthetic_video: Path) -> None:
    source = FileVideoSource(synthetic_video, loop=True)
    with source:
        frames = [source.read() for _ in range(8)]

    assert all(frame is not None for frame in frames)
    # Indices keep counting across the loop boundary so downstream state stays valid.
    assert [frame.index for frame in frames if frame] == list(range(8))


def test_non_looping_file_source_stops_and_stays_stopped(synthetic_video: Path) -> None:
    with FileVideoSource(synthetic_video) as source:
        for _ in range(5):
            assert source.read() is not None
        assert source.read() is None
        assert source.read() is None


def test_reading_before_open_fails_loudly(synthetic_video: Path) -> None:
    source = FileVideoSource(synthetic_video)

    with pytest.raises(VideoSourceError, match="not open"):
        source.read()


def test_metadata_before_open_fails_loudly(synthetic_video: Path) -> None:
    source = FileVideoSource(synthetic_video)

    with pytest.raises(VideoSourceError, match="not open"):
        _ = source.metadata


def test_context_manager_releases_capture_even_on_error(synthetic_video: Path) -> None:
    source = FileVideoSource(synthetic_video)

    with pytest.raises(RuntimeError), source:
        source.read()
        raise RuntimeError("boom")

    with pytest.raises(VideoSourceError, match="not open"):
        source.read()


def test_missing_file_is_reported_at_construction(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError, match="not found"):
        FileVideoSource(tmp_path / "nope.mp4")


def test_factory_selects_webcam_for_integer_specifier() -> None:
    source = create_video_source("1")

    assert isinstance(source, WebcamVideoSource)


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://camera.local/stream1",
        "RTSP://camera.local/stream1",
        "http://example.com/live.m3u8",
        "udp://239.0.0.1:1234",
    ],
)
def test_factory_selects_stream_for_url_specifiers(url: str) -> None:
    source = create_video_source(url)

    assert isinstance(source, StreamVideoSource)


def test_factory_selects_file_for_paths(synthetic_video: Path) -> None:
    source = create_video_source(str(synthetic_video))

    assert isinstance(source, FileVideoSource)


def test_factory_rejects_empty_specifier() -> None:
    with pytest.raises(VideoSourceError, match="must not be empty"):
        create_video_source("   ")


def test_factory_reports_missing_file_path() -> None:
    with pytest.raises(VideoSourceError, match="not found"):
        create_video_source("data/input/definitely-missing.mp4")
