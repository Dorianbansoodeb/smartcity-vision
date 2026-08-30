"""Frame-processing pipeline.

:class:`VideoProcessor` is the only place that knows the order of operations for
a run: pull a frame, detect, annotate, write, record timings. Analytics modules
added in later phases plug in here rather than into the detector or the renderer.
"""

from __future__ import annotations

import statistics
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from smartcity_vision.detection.detector import DetectionResult, YoloDetector
from smartcity_vision.exceptions import VideoSourceError
from smartcity_vision.utils.config import AppConfig
from smartcity_vision.utils.logging import get_logger
from smartcity_vision.video.source import (
    Frame,
    VideoMetadata,
    VideoSource,
    create_video_source,
)
from smartcity_vision.visualization.renderer import FrameRenderer

logger = get_logger(__name__)

_FPS_WINDOW = 30


def create_source_from_config(config: AppConfig) -> VideoSource:
    """Build the video source described by ``config``.

    Kept separate from :class:`VideoProcessor` so a caller can construct the
    source first and fail on a bad path before paying to load model weights.
    """
    return create_video_source(
        config.video.source,
        loop=config.video.loop,
        fallback_fps=config.output.fallback_fps,
    )


@dataclass(slots=True)
class ProcessingStats:
    """Measured performance and detection totals for one run.

    Every field is populated from an actual run; nothing here is estimated.
    """

    frames_read: int = 0
    frames_processed: int = 0
    frames_skipped: int = 0
    total_detections: int = 0
    elapsed_seconds: float = 0.0
    detections_by_class: dict[str, int] = field(default_factory=dict)
    inference_ms_samples: list[float] = field(default_factory=list, repr=False)
    output_video: Path | None = None
    device: str = "unknown"
    interrupted: bool = False

    @property
    def pipeline_fps(self) -> float:
        """End-to-end throughput: processed frames per wall-clock second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds

    @property
    def avg_inference_ms(self) -> float:
        """Mean model inference latency per processed frame."""
        if not self.inference_ms_samples:
            return 0.0
        return statistics.fmean(self.inference_ms_samples)

    @property
    def p95_inference_ms(self) -> float:
        """95th-percentile inference latency, the number that matters for SLOs."""
        if not self.inference_ms_samples:
            return 0.0
        ordered = sorted(self.inference_ms_samples)
        index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[index]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of the run."""
        return {
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_skipped": self.frames_skipped,
            "total_detections": self.total_detections,
            "detections_by_class": dict(sorted(self.detections_by_class.items())),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "pipeline_fps": round(self.pipeline_fps, 2),
            "avg_inference_ms": round(self.avg_inference_ms, 2),
            "p95_inference_ms": round(self.p95_inference_ms, 2),
            "device": self.device,
            "output_video": str(self.output_video) if self.output_video else None,
            "interrupted": self.interrupted,
        }


class VideoProcessor:
    """Runs detection over a video source and writes an annotated result."""

    def __init__(
        self,
        config: AppConfig,
        detector: YoloDetector,
        renderer: FrameRenderer | None = None,
        source: VideoSource | None = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            config: Validated application configuration.
            detector: Detector to reuse for every frame.
            renderer: Overlay renderer; built from config when omitted.
            source: Video source to consume; built from config when omitted.
        """
        self._config = config
        self._detector = detector
        self._renderer = renderer or FrameRenderer(config.visualization)
        self._source = source or create_source_from_config(config)
        self._recent_frame_durations: deque[float] = deque(maxlen=_FPS_WINDOW)

    def run(self) -> ProcessingStats:
        """Process the configured source end to end.

        The source is consumed once; construct a new processor to run again.

        Returns:
            Measured statistics for the run. ``Ctrl-C`` stops early and still
            returns the statistics gathered so far with ``interrupted`` set.
        """
        video_config = self._config.video
        source = self._source
        stats = ProcessingStats(device=self._detector.device)
        class_counter: Counter[str] = Counter()
        writer: cv2.VideoWriter | None = None
        display_enabled = video_config.display

        with source:
            metadata = source.metadata
            if self._config.model.warmup:
                self._detector.warmup(metadata.width, metadata.height)
            started = perf_counter()
            try:
                for frame in source:
                    stats.frames_read += 1
                    if self._should_skip(frame):
                        stats.frames_skipped += 1
                        continue

                    frame_started = perf_counter()
                    result = self._detector.detect(frame)
                    annotated = self._annotate(frame, result)

                    if writer is None and self._config.output.write_annotated_video:
                        writer = self._create_writer(metadata, annotated)
                        stats.output_video = self._annotated_video_path()
                    if writer is not None:
                        writer.write(annotated)

                    if display_enabled:
                        display_enabled = self._show(annotated)
                        if not display_enabled and video_config.display:
                            logger.info("Display closed; continuing headless")

                    stats.frames_processed += 1
                    stats.total_detections += len(result)
                    stats.inference_ms_samples.append(result.inference_ms)
                    class_counter.update(detection.class_name for detection in result.detections)
                    self._recent_frame_durations.append(perf_counter() - frame_started)

                    self._log_progress(stats, metadata)
                    if self._reached_frame_limit(stats):
                        logger.info("Reached max_frames=%d; stopping", video_config.max_frames)
                        break
            except KeyboardInterrupt:
                stats.interrupted = True
                logger.warning("Interrupted after %d processed frames", stats.frames_processed)
            finally:
                stats.elapsed_seconds = perf_counter() - started
                if writer is not None:
                    writer.release()
                if video_config.display:
                    cv2.destroyAllWindows()

        stats.detections_by_class = dict(class_counter)
        self._log_summary(stats)
        return stats

    def _annotate(self, frame: Frame, result: DetectionResult) -> np.ndarray:
        """Return an annotated copy of ``frame``, leaving the original untouched."""
        annotated = frame.image.copy()
        self._renderer.draw_detections(annotated, result.detections)
        self._renderer.draw_hud(annotated, self._hud_lines(frame, result))
        return annotated

    def _hud_lines(self, frame: Frame, result: DetectionResult) -> list[str]:
        """Build the status-panel text for a frame."""
        lines = [
            f"Frame {frame.index}  t={frame.timestamp:.2f}s",
            f"FPS {self._rolling_fps():.1f}  ({self._detector.device})",
            f"Objects {len(result)}",
        ]
        by_class = Counter(detection.class_name for detection in result.detections)
        if by_class:
            lines.append("  ".join(f"{name}:{count}" for name, count in sorted(by_class.items())))
        return lines

    def _rolling_fps(self) -> float:
        """Throughput over the recent window, so the HUD reacts to slowdowns."""
        if not self._recent_frame_durations:
            return 0.0
        mean_duration = statistics.fmean(self._recent_frame_durations)
        return 1.0 / mean_duration if mean_duration > 0 else 0.0

    def _should_skip(self, frame: Frame) -> bool:
        """Whether ``frame`` is dropped by the frame-skip setting."""
        stride = self._config.video.frame_skip + 1
        return stride > 1 and frame.index % stride != 0

    def _reached_frame_limit(self, stats: ProcessingStats) -> bool:
        """Whether the configured processed-frame budget is used up."""
        limit = self._config.video.max_frames
        return limit is not None and stats.frames_processed >= limit

    def _annotated_video_path(self) -> Path:
        """Resolve the output video path from config."""
        output = self._config.output
        return output.directory / output.annotated_video_name

    def _create_writer(self, metadata: VideoMetadata, frame: np.ndarray) -> cv2.VideoWriter:
        """Open the annotated-video writer sized to the frames actually produced.

        With frame skipping the output holds fewer frames, so its frame rate is
        divided by the stride to keep playback duration true to the source.
        """
        path = self._annotated_video_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        height, width = frame.shape[:2]
        stride = self._config.video.frame_skip + 1
        fps = max(metadata.fps / stride, 1.0)
        writer = cv2.VideoWriter(
            str(path),
            _fourcc(self._config.output.video_codec),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise VideoSourceError(
                f"Could not open video writer for {path} "
                f"with codec {self._config.output.video_codec!r}"
            )
        logger.info("Writing annotated video to %s (%dx%d @ %.2f fps)", path, width, height, fps)
        return writer

    @staticmethod
    def _show(annotated: np.ndarray) -> bool:
        """Show a preview window; return ``False`` if display is unavailable or quit."""
        try:
            cv2.imshow("SmartCity Vision", annotated)
            return cv2.waitKey(1) & 0xFF != ord("q")
        except cv2.error as exc:
            logger.warning("Preview window unavailable (%s)", exc)
            return False

    def _log_progress(self, stats: ProcessingStats, metadata: VideoMetadata) -> None:
        """Emit a periodic progress line."""
        interval = self._config.logging.progress_every_frames
        if stats.frames_processed % interval != 0:
            return
        if metadata.frame_count:
            share = 100.0 * stats.frames_read / metadata.frame_count
            logger.info(
                "Processed %d/%d frames (%.1f%%) | %.1f fps | %d detections",
                stats.frames_processed,
                metadata.frame_count,
                share,
                self._rolling_fps(),
                stats.total_detections,
            )
        else:
            logger.info(
                "Processed %d frames | %.1f fps | %d detections",
                stats.frames_processed,
                self._rolling_fps(),
                stats.total_detections,
            )

    @staticmethod
    def _log_summary(stats: ProcessingStats) -> None:
        """Log the measured end-of-run numbers."""
        logger.info(
            "Run complete: %d processed (%d skipped) in %.2fs | %.2f fps | "
            "inference avg %.1f ms, p95 %.1f ms | %d detections",
            stats.frames_processed,
            stats.frames_skipped,
            stats.elapsed_seconds,
            stats.pipeline_fps,
            stats.avg_inference_ms,
            stats.p95_inference_ms,
            stats.total_detections,
        )


def _fourcc(codec: str) -> int:
    """Return the FourCC code for ``codec`` across OpenCV 4 and 5 APIs."""
    legacy = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(legacy):
        return int(legacy(*codec))
    return int(cv2.VideoWriter.fourcc(*codec))


__all__ = ["ProcessingStats", "VideoProcessor"]
