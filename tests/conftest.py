"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Write a short deterministic MP4 and return its path.

    Frame ``i`` is filled with the constant grey value ``10 * i``, which lets
    tests assert that frames arrive in order and unmodified.
    """
    path = tmp_path / "synthetic.mp4"
    width, height, fps, frame_count = 64, 48, 10.0, 5
    fourcc = getattr(cv2, "VideoWriter_fourcc", None)
    code = fourcc(*"mp4v") if callable(fourcc) else cv2.VideoWriter.fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(path), code, fps, (width, height))
    assert writer.isOpened(), "OpenCV could not open an mp4v writer"
    for index in range(frame_count):
        writer.write(np.full((height, width, 3), 10 * index, dtype=np.uint8))
    writer.release()

    assert path.is_file() and path.stat().st_size > 0
    return path
