"""Video source abstraction.

The rest of the pipeline consumes :class:`Frame` objects and never touches
:mod:`cv2` capture handles, so a file, a webcam, and an RTSP stream are
interchangeable. Files carry deterministic timestamps derived from their frame
rate; live sources are stamped from a monotonic clock, because a live stream has
no meaningful "frame N of M".
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final

import cv2
import numpy as np

from smartcity_vision.exceptions import VideoSourceError
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)

_STREAM_SCHEMES: Final = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
_DEFAULT_FALLBACK_FPS: Final = 25.0


@dataclass(frozen=True, slots=True)
class Frame:
    """A single decoded frame.

    Attributes:
        index: Zero-based counter of frames read from the source.
        timestamp: Seconds since the start of the stream.
        image: BGR image of shape ``(height, width, 3)``.
    """

    index: int
    timestamp: float
    image: np.ndarray


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Properties of an opened source.

    Attributes:
        name: Human-readable identifier of the source.
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frame rate; falls back to a configured default when unreported.
        frame_count: Total frames, or ``None`` for live sources.
        is_live: Whether the source is unbounded (webcam or network stream).
    """

    name: str
    width: int
    height: int
    fps: float
    frame_count: int | None
    is_live: bool


class VideoSource(AbstractContextManager["VideoSource"], ABC):
    """Iterable, closeable source of :class:`Frame` objects."""

    @abstractmethod
    def open(self) -> None:
        """Acquire the underlying capture. Idempotent."""

    @abstractmethod
    def close(self) -> None:
        """Release the underlying capture. Idempotent."""

    @abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the source is exhausted."""

    @property
    @abstractmethod
    def metadata(self) -> VideoMetadata:
        """Metadata of the opened source."""

    def __enter__(self) -> VideoSource:
        """Open the source on context entry."""
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Always release the capture, even if the body raised."""
        self.close()

    def __iter__(self) -> Iterator[Frame]:
        """Yield frames until the source is exhausted."""
        self.open()
        while True:
            frame = self.read()
            if frame is None:
                return
            yield frame


class _OpenCVVideoSource(VideoSource):
    """Shared OpenCV capture behaviour for files, webcams, and streams."""

    def __init__(self, name: str, fallback_fps: float = _DEFAULT_FALLBACK_FPS) -> None:
        self._name = name
        self._fallback_fps = fallback_fps
        self._capture: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None
        self._frame_index = 0
        self._start_time = 0.0

    @property
    @abstractmethod
    def _capture_argument(self) -> str | int:
        """Value handed to :class:`cv2.VideoCapture`."""

    @property
    @abstractmethod
    def _is_live(self) -> bool:
        """Whether this source is unbounded."""

    def open(self) -> None:
        """Open the capture and read its properties."""
        if self._capture is not None:
            return

        capture = cv2.VideoCapture(self._capture_argument)
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(f"Could not open video source: {self._name}")

        self._capture = capture
        self._metadata = self._read_metadata(capture)
        self._frame_index = 0
        self._start_time = time.monotonic()
        logger.info(
            "Opened %s (%dx%d @ %.2f fps, frames=%s)",
            self._name,
            self._metadata.width,
            self._metadata.height,
            self._metadata.fps,
            self._metadata.frame_count if self._metadata.frame_count else "unknown",
        )

    def close(self) -> None:
        """Release the capture if it is open."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.debug("Closed %s after %d frames", self._name, self._frame_index)

    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` at end of stream."""
        capture = self._require_capture()
        ok, image = capture.read()
        if not ok or image is None:
            return self._on_read_failure()

        frame = Frame(
            index=self._frame_index,
            timestamp=self._timestamp_for(self._frame_index),
            image=image,
        )
        self._frame_index += 1
        return frame

    @property
    def metadata(self) -> VideoMetadata:
        """Metadata of the opened source."""
        if self._metadata is None:
            raise VideoSourceError(f"Source {self._name} is not open; call open() first")
        return self._metadata

    def _on_read_failure(self) -> Frame | None:
        """Handle a failed read. Subclasses may retry or restart instead."""
        return None

    def _timestamp_for(self, frame_index: int) -> float:
        """Return the timestamp to attach to ``frame_index``."""
        if self._is_live:
            return time.monotonic() - self._start_time
        return frame_index / self.metadata.fps

    def _require_capture(self) -> cv2.VideoCapture:
        """Return the open capture or fail loudly."""
        if self._capture is None:
            raise VideoSourceError(f"Source {self._name} is not open; call open() first")
        return self._capture

    def _read_metadata(self, capture: cv2.VideoCapture) -> VideoMetadata:
        """Extract metadata, substituting sane values for unreported properties."""
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not fps or fps <= 0 or not np.isfinite(fps):
            logger.warning(
                "%s did not report a frame rate; assuming %.2f fps",
                self._name,
                self._fallback_fps,
            )
            fps = self._fallback_fps

        raw_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = None if self._is_live or raw_count <= 0 else raw_count

        return VideoMetadata(
            name=self._name,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frame_count=frame_count,
            is_live=self._is_live,
        )


class FileVideoSource(_OpenCVVideoSource):
    """Frames from a video file on disk."""

    def __init__(
        self,
        path: Path | str,
        loop: bool = False,
        fallback_fps: float = _DEFAULT_FALLBACK_FPS,
    ) -> None:
        """Initialise a file source.

        Args:
            path: Path to the video file.
            loop: Restart from the first frame at end of file instead of stopping.
            fallback_fps: Frame rate assumed when the container reports none.

        Raises:
            VideoSourceError: If the file does not exist.
        """
        self._path = Path(path).expanduser()
        if not self._path.is_file():
            raise VideoSourceError(f"Video file not found: {self._path}")
        super().__init__(name=str(self._path), fallback_fps=fallback_fps)
        self._loop = loop

    @property
    def _capture_argument(self) -> str:
        return str(self._path)

    @property
    def _is_live(self) -> bool:
        return False

    def _on_read_failure(self) -> Frame | None:
        """Rewind and continue when looping, otherwise end the stream."""
        if not self._loop:
            return None

        capture = self._require_capture()
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, image = capture.read()
        if not ok or image is None:
            logger.warning("Could not restart %s for looping; ending stream", self._name)
            return None

        logger.debug("Looping %s after %d frames", self._name, self._frame_index)
        frame = Frame(
            index=self._frame_index,
            timestamp=self._timestamp_for(self._frame_index),
            image=image,
        )
        self._frame_index += 1
        return frame


class WebcamVideoSource(_OpenCVVideoSource):
    """Frames from a locally attached camera."""

    def __init__(self, device_index: int = 0, fallback_fps: float = _DEFAULT_FALLBACK_FPS) -> None:
        """Initialise a webcam source.

        Args:
            device_index: OpenCV camera index, usually ``0`` for the built-in camera.
            fallback_fps: Frame rate assumed when the driver reports none.
        """
        super().__init__(name=f"webcam:{device_index}", fallback_fps=fallback_fps)
        self._device_index = device_index

    @property
    def _capture_argument(self) -> int:
        return self._device_index

    @property
    def _is_live(self) -> bool:
        return True


class StreamVideoSource(_OpenCVVideoSource):
    """Frames from a network stream such as RTSP or HTTP(S)."""

    def __init__(self, url: str, fallback_fps: float = _DEFAULT_FALLBACK_FPS) -> None:
        """Initialise a network stream source.

        Args:
            url: Stream URL, e.g. ``rtsp://camera.local/stream1``.
            fallback_fps: Frame rate assumed when the stream reports none.
        """
        super().__init__(name=url, fallback_fps=fallback_fps)
        self._url = url

    @property
    def _capture_argument(self) -> str:
        return self._url

    @property
    def _is_live(self) -> bool:
        return True


def create_video_source(
    specifier: str,
    loop: bool = False,
    fallback_fps: float = _DEFAULT_FALLBACK_FPS,
) -> VideoSource:
    """Build the right :class:`VideoSource` for a source string.

    Dispatch rules: an integer selects a webcam, a known URL scheme selects a
    network stream, anything else is treated as a filesystem path.

    Args:
        specifier: ``"0"``, ``"rtsp://host/stream"``, or ``"data/input/clip.mp4"``.
        loop: Only meaningful for files; restart at end of file.
        fallback_fps: Frame rate assumed when the source reports none.

    Returns:
        An unopened video source.

    Raises:
        VideoSourceError: If ``specifier`` is empty or names a missing file.
    """
    candidate = specifier.strip()
    if not candidate:
        raise VideoSourceError("Video source specifier must not be empty")

    if candidate.isdigit():
        return WebcamVideoSource(device_index=int(candidate), fallback_fps=fallback_fps)

    if candidate.lower().startswith(_STREAM_SCHEMES):
        return StreamVideoSource(url=candidate, fallback_fps=fallback_fps)

    return FileVideoSource(path=candidate, loop=loop, fallback_fps=fallback_fps)


__all__ = [
    "FileVideoSource",
    "Frame",
    "StreamVideoSource",
    "VideoMetadata",
    "VideoSource",
    "WebcamVideoSource",
    "create_video_source",
]
