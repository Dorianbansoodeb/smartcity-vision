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

from smartcity_vision.analytics.counter import CountingSummary, UniqueObjectCounter
from smartcity_vision.analytics.density import DensityEstimator
from smartcity_vision.analytics.line_crossing import CrossingSummary, LineCrossingDetector
from smartcity_vision.analytics.queue import QueueEstimator
from smartcity_vision.analytics.speed import SpeedEstimator
from smartcity_vision.analytics.trajectories import TrajectoryStore
from smartcity_vision.analytics.zones import ZoneMonitor, ZoneSummary
from smartcity_vision.database.repository import DetectionRow, MetricRow
from smartcity_vision.detection.detector import DetectionResult, YoloDetector
from smartcity_vision.exceptions import VideoSourceError
from smartcity_vision.monitoring.drift import DriftDetector
from smartcity_vision.privacy.anonymizer import FrameAnonymizer
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
    tracker: str | None = None
    counting: CountingSummary | None = None
    crossings: CrossingSummary | None = None
    zones: ZoneSummary | None = None
    density: dict[str, Any] | None = None
    queue: dict[str, Any] | None = None
    speed: dict[str, Any] | None = None
    detection_rows: list[DetectionRow] = field(default_factory=list, repr=False)
    metric_rows: list[MetricRow] = field(default_factory=list, repr=False)
    run_id: str | None = None
    drift: dict[str, Any] | None = None
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
            "tracker": self.tracker,
            "unique_counts": self.counting.as_dict() if self.counting else None,
            "crossings": self.crossings.as_dict() if self.crossings else None,
            "zones": self.zones.as_dict() if self.zones else None,
            "density": self.density,
            "queue": self.queue,
            "speed": self.speed,
            "run_id": self.run_id,
            "drift": self.drift,
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
        counter: UniqueObjectCounter | None = None,
    ) -> None:
        """Initialise the pipeline.

        Args:
            config: Validated application configuration.
            detector: Detector or tracker to reuse for every frame.
            renderer: Overlay renderer; built from config when omitted.
            source: Video source to consume; built from config when omitted.
            counter: Unique-object counter. Omitted builds one from config when
                tracking is enabled; without tracking there are no identities to
                count, so counting is skipped.
        """
        self._config = config
        self._detector = detector
        self._renderer = renderer or FrameRenderer(config.visualization)
        self._source = source or create_source_from_config(config)
        self._counter = counter or self._default_counter(config)
        self._crossings = LineCrossingDetector(config.analytics.lines)
        self._zones = ZoneMonitor(config.analytics.zones)
        self._trajectories = TrajectoryStore(config.analytics.trajectories)
        self._density = DensityEstimator(
            config.analytics.density, _region_polygon(config, config.analytics.density.region)
        )
        self._queue = QueueEstimator(config.analytics.queue)
        self._speed = SpeedEstimator(config.analytics.speed)
        self._anonymizer = FrameAnonymizer(config.privacy)
        self._drift = DriftDetector(config.monitoring)
        self._recent_frame_durations: deque[float] = deque(maxlen=_FPS_WINDOW)

    @staticmethod
    def _default_counter(config: AppConfig) -> UniqueObjectCounter | None:
        """Build a counter when tracking can supply identities, else ``None``."""
        if not config.tracking.enabled:
            logger.info("Tracking disabled; unique object counts will not be produced")
            return None
        return UniqueObjectCounter(config.analytics.counting)

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
                    if self._counter is not None:
                        self._counter.update(result)
                    self._crossings.update(result)
                    self._zones.update(result)
                    self._trajectories.update(result)
                    self._density.update(result)
                    self._queue.update(result, self._trajectories)
                    self._speed.update(result, self._trajectories)
                    self._drift.update(result)
                    annotated = self._annotate(frame, result)
                    if self._anonymizer.enabled:
                        annotated, _ = self._anonymizer.anonymize(
                            annotated, list(result.detections)
                        )

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
                    stats.detection_rows.extend(
                        DetectionRow(
                            frame_index=result.frame_index,
                            timestamp=result.timestamp,
                            detection=detection,
                        )
                        for detection in result.detections
                    )
                    density = self._density.current
                    queue = self._queue.current
                    speeds = self._speed.current
                    stats.metric_rows.append(
                        MetricRow(
                            frame_index=result.frame_index,
                            timestamp=result.timestamp,
                            vehicles_in_frame=0 if density is None else density.vehicles_in_frame,
                            vehicles_in_region=0 if density is None else density.vehicles_in_region,
                            congestion="LOW" if density is None else density.congestion,
                            queued_vehicles=0 if queue is None else queue.queued_vehicles,
                            queue_length_px=0.0 if queue is None else queue.length_px,
                            mean_speed_px_s=(
                                None
                                if not speeds
                                else sum(item.speed_px_s for item in speeds) / len(speeds)
                            ),
                        )
                    )
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
        stats.tracker = getattr(self._detector, "tracker_name", None)
        if self._counter is not None:
            stats.counting = self._counter.summary()
        stats.crossings = self._crossings.summary()
        stats.zones = self._zones.summary()
        stats.density = self._density.as_dict()
        stats.queue = self._queue.as_dict()
        stats.speed = self._speed.as_dict()
        stats.drift = None if self._drift.latest is None else self._drift.latest.as_dict()
        self._log_summary(stats)
        return stats

    def _annotate(self, frame: Frame, result: DetectionResult) -> np.ndarray:
        """Return an annotated copy of ``frame``, leaving the original untouched."""
        annotated = frame.image.copy()
        self._renderer.draw_zones(annotated, self._config.analytics.zones)
        self._renderer.draw_lines(annotated, self._config.analytics.lines)
        self._renderer.draw_trajectories(annotated, self._trajectories.active_trails())
        self._renderer.draw_detections(annotated, result.detections)
        self._renderer.draw_hud(annotated, self._hud_lines(frame, result))
        return annotated

    def _hud_lines(self, frame: Frame, result: DetectionResult) -> list[str]:
        """Build the status-panel text for a frame."""
        lines = [
            f"Frame {frame.index}  t={frame.timestamp:.2f}s",
            f"FPS {self._rolling_fps():.1f}  ({self._detector.device})",
            f"In frame: {len(result)}",
        ]
        by_class = Counter(detection.class_name for detection in result.detections)
        if by_class:
            lines.append("  ".join(f"{name}:{count}" for name, count in sorted(by_class.items())))

        if self._counter is not None:
            summary = self._counter.summary()
            lines.append(f"Unique total: {summary.total}")
            if summary.counts_by_class:
                lines.append(
                    "  ".join(f"{name}:{count}" for name, count in summary.counts_by_class.items())
                )

        crossings = self._crossings.summary()
        if crossings.events:
            lines.append(f"Crossings: {len(crossings.events)}")
        zone_bits = [
            f"{name}:{occ.occupants}"
            for name, occ in self._zones.summary().occupancy.items()
            if occ.occupants
        ]
        if zone_bits:
            lines.append("Zone " + "  ".join(zone_bits))
        density = self._density.current
        if density is not None:
            lines.append(f"Density {density.vehicles_in_region}  {density.congestion}")
        queue = self._queue.current
        if queue is not None and queue.queued_vehicles:
            lines.append(f"Queue {queue.queued_vehicles}  {queue.length_px:.0f}px")
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
        if stats.counting is not None:
            logger.info(
                "Unique objects: %d total (%s) from %d tracks observed "
                "(%d discarded as too short-lived, %d still unconfirmed)",
                stats.counting.total,
                ", ".join(
                    f"{name} {count}" for name, count in stats.counting.counts_by_class.items()
                )
                or "none",
                stats.counting.tracks_observed,
                stats.counting.discarded_tracks,
                stats.counting.pending_tracks,
            )
        if stats.crossings is not None and stats.crossings.events:
            logger.info(
                "Line crossings: %d (%s)",
                len(stats.crossings.events),
                ", ".join(
                    f"{name} {count}" for name, count in stats.crossings.counts_by_line.items()
                ),
            )
        if stats.zones is not None and (stats.zones.enters or stats.zones.exits):
            logger.info(
                "Zone events: %d enter, %d exit",
                stats.zones.enters,
                stats.zones.exits,
            )
        if stats.density is not None and stats.density.get("current"):
            logger.info(
                "Peak congestion: %s (last rolling avg %.2f vehicles)",
                stats.density["peak_congestion"],
                stats.density["current"]["rolling_average"],
            )
        if stats.speed is not None and stats.speed.get("mean_speed_px_s") is not None:
            logger.info(
                "Mean estimated speed: %.1f px/s%s",
                stats.speed["mean_speed_px_s"],
                "" if not stats.speed["calibrated"] else f" ({stats.speed['mean_speed_kmh']} km/h)",
            )


def _region_polygon(config: AppConfig, region_name: str) -> tuple[tuple[float, float], ...] | None:
    """Resolve a density region name to a polygon, or ``None`` for the whole frame."""
    if not region_name:
        return None
    for zone in config.analytics.zones:
        if zone.name == region_name:
            return zone.polygon
    logger.warning("Density region %r is not a configured zone; using the whole frame", region_name)
    return None


def _fourcc(codec: str) -> int:
    """Return the FourCC code for ``codec`` across OpenCV 4 and 5 APIs."""
    legacy = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(legacy):
        return int(legacy(*codec))
    return int(cv2.VideoWriter.fourcc(*codec))


__all__ = ["ProcessingStats", "VideoProcessor"]
