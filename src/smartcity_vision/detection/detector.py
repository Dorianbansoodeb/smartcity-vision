"""YOLOv8 detection.

The detector owns the model for the whole lifetime of a run: weights are loaded
once and every frame reuses them, which is the difference between a few frames
per second and a few seconds per frame. Ultralytics results are converted
immediately into plain :class:`Detection` objects so no other module depends on
the Ultralytics result API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset

from smartcity_vision.exceptions import ConfigError, DetectionError
from smartcity_vision.utils.config import ModelConfig
from smartcity_vision.utils.logging import get_logger
from smartcity_vision.video.source import Frame

logger = get_logger(__name__)

BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object in one frame.

    Attributes:
        class_id: Model class index.
        class_name: Human-readable class name, e.g. ``"car"``.
        confidence: Detection confidence in ``[0, 1]``.
        bbox: Pixel bounding box as ``(x1, y1, x2, y2)``.
        track_id: Identity assigned by the tracker, stable across frames.
            ``None`` when running detection without tracking, or for a detection
            the tracker has not yet confirmed into a track.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    track_id: int | None = None

    @property
    def center(self) -> tuple[float, float]:
        """Bounding-box centre as ``(x, y)`` pixels."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width(self) -> float:
        """Bounding-box width in pixels."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Bounding-box height in pixels."""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """Bounding-box area in square pixels."""
        return self.width * self.height

    def int_bbox(self) -> tuple[int, int, int, int]:
        """Bounding box rounded to integer pixels, for drawing."""
        x1, y1, x2, y2 = self.bbox
        return (round(x1), round(y1), round(x2), round(y2))


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """All detections for a single frame, plus how long inference took.

    Attributes:
        frame_index: Index of the source frame.
        timestamp: Seconds since the start of the stream.
        detections: Detections that passed the configured thresholds.
        inference_ms: Wall-clock inference time for this frame, in milliseconds.
    """

    frame_index: int
    timestamp: float
    detections: tuple[Detection, ...]
    inference_ms: float

    def __len__(self) -> int:
        """Number of detections in this frame."""
        return len(self.detections)

    @property
    def tracked(self) -> tuple[Detection, ...]:
        """Detections that carry a track identity."""
        return tuple(item for item in self.detections if item.track_id is not None)


def resolve_device(requested: str) -> str:
    """Map a requested device onto one that is actually available.

    ``"auto"`` prefers CUDA, then Apple Silicon MPS, then CPU. An explicit
    request for unavailable hardware falls back to CPU with a warning rather
    than crashing mid-run.

    Args:
        requested: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns:
        The device string to hand to Ultralytics.
    """
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if requested == "auto":
        if cuda_available:
            return "cuda"
        if mps_available:
            return "mps"
        return "cpu"

    if requested == "cuda" and not cuda_available:
        logger.warning("CUDA requested but unavailable; falling back to CPU")
        return "cpu"
    if requested == "mps" and not mps_available:
        logger.warning("MPS requested but unavailable; falling back to CPU")
        return "cpu"
    return requested


class YoloDetector:
    """Runs YOLOv8 inference on frames and returns typed detections."""

    def __init__(self, config: ModelConfig) -> None:
        """Load weights and resolve the target classes.

        Args:
            config: Validated model configuration.

        Raises:
            DetectionError: If the weights cannot be loaded.
            ConfigError: If a configured target class is not in the model.
        """
        self._config = config
        self._device = resolve_device(config.device)
        self._model = self._load_model(config.weights)
        self._class_names: dict[int, str] = {
            int(index): str(name) for index, name in self._model.names.items()
        }
        self._target_class_ids = self._resolve_target_class_ids(config.target_classes)

        logger.info(
            "YOLO ready: weights=%s device=%s classes=%s conf=%.2f iou=%.2f imgsz=%d",
            config.weights.name,
            self._device,
            ",".join(config.target_classes),
            config.confidence_threshold,
            config.iou_threshold,
            config.image_size,
        )

    @property
    def device(self) -> str:
        """Device inference actually runs on."""
        return self._device

    @property
    def class_names(self) -> dict[int, str]:
        """Class index to name mapping exposed by the loaded model."""
        return dict(self._class_names)

    def warmup(self, width: int, height: int) -> float:
        """Run one throwaway inference to absorb lazy device initialisation.

        The first inference on CUDA or MPS pays for kernel compilation and
        allocator setup, which is often two orders of magnitude slower than
        steady state. Doing it here keeps that cost out of the measured
        per-frame latency instead of silently inflating the average.

        Args:
            width: Frame width the run will use, in pixels.
            height: Frame height the run will use, in pixels.

        Returns:
            The warmup inference time in milliseconds.
        """
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        started = perf_counter()
        # Deliberately plain prediction: a warmup frame must not enter tracker state.
        self._model.predict(source=blank, **self._inference_kwargs())
        elapsed_ms = (perf_counter() - started) * 1000.0
        logger.info("Warmup inference took %.1f ms on %s", elapsed_ms, self._device)
        return elapsed_ms

    def detect(self, frame: Frame) -> DetectionResult:
        """Run inference on one frame.

        Args:
            frame: Frame to analyse.

        Returns:
            Detections above the confidence threshold, restricted to the
            configured target classes.

        Raises:
            DetectionError: If the model raises during inference.
        """
        started = perf_counter()
        try:
            results = self._run(frame.image)
        except Exception as exc:  # noqa: BLE001 - surfaced as a package error
            raise DetectionError(f"Inference failed on frame {frame.index}: {exc}") from exc
        inference_ms = (perf_counter() - started) * 1000.0

        detections = self._parse_results(results)
        return DetectionResult(
            frame_index=frame.index,
            timestamp=frame.timestamp,
            detections=detections,
            inference_ms=inference_ms,
        )

    def _run(self, image: np.ndarray) -> list:
        """Invoke the model on one image.

        Overridden by :class:`~smartcity_vision.detection.tracker.YoloTracker` to
        run tracking instead of stateless prediction.
        """
        return self._model.predict(source=image, **self._inference_kwargs())

    def _inference_kwargs(self) -> dict[str, object]:
        """Keyword arguments shared by prediction and tracking calls."""
        return {
            "conf": self._config.confidence_threshold,
            "iou": self._config.iou_threshold,
            "imgsz": self._config.image_size,
            "max_det": self._config.max_detections,
            "classes": list(self._target_class_ids),
            "device": self._device,
            "verbose": False,
        }

    def _parse_results(self, results: list) -> tuple[Detection, ...]:
        """Convert Ultralytics results into :class:`Detection` objects."""
        if not results:
            return ()

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return ()

        xyxy = np.asarray(boxes.xyxy.cpu(), dtype=float)
        confidences = np.asarray(boxes.conf.cpu(), dtype=float)
        class_ids = np.asarray(boxes.cls.cpu(), dtype=int)
        # boxes.id is absent for plain prediction and None until the tracker
        # confirms its first track, so both cases collapse to "no identities".
        raw_ids = getattr(boxes, "id", None)
        track_ids = (
            [None] * len(class_ids)
            if raw_ids is None
            else [int(value) for value in np.asarray(raw_ids.cpu(), dtype=int)]
        )

        return tuple(
            Detection(
                class_id=int(class_id),
                class_name=self._class_names.get(int(class_id), str(class_id)),
                confidence=float(confidence),
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                track_id=track_id,
            )
            for box, confidence, class_id, track_id in zip(
                xyxy, confidences, class_ids, track_ids, strict=True
            )
        )

    def _load_model(self, weights: Path) -> YOLO:
        """Load weights, downloading a pretrained checkpoint on first use."""
        path = weights if weights.is_file() else self._download_pretrained(weights)
        logger.debug("Loading weights from %s", path)
        try:
            return YOLO(str(path))
        except Exception as exc:  # noqa: BLE001 - surfaced as a package error
            raise DetectionError(f"Could not load weights {path}: {exc}") from exc

    @staticmethod
    def _download_pretrained(weights: Path) -> Path:
        """Fetch the official checkpoint named by ``weights`` to that exact path.

        A missing local file is treated as a request for the equivalently named
        pretrained checkpoint. Downloading straight to the configured path keeps
        the weight file inside the project (rather than the working directory)
        and makes later runs offline and reproducible.
        """
        logger.info(
            "Weights %s not found locally; downloading pretrained %s", weights, weights.name
        )
        weights.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = Path(attempt_download_asset(weights))
        except Exception as exc:  # noqa: BLE001 - surfaced as a package error
            raise DetectionError(
                f"Could not download pretrained weights {weights.name!r}: {exc}"
            ) from exc

        if not downloaded.is_file():
            raise DetectionError(f"Pretrained weights {weights.name!r} were not downloaded")
        logger.info("Pretrained weights ready at %s", downloaded)
        return downloaded

    def _resolve_target_class_ids(self, target_classes: tuple[str, ...]) -> tuple[int, ...]:
        """Map configured class names to model class indices."""
        by_name = {name.lower(): index for index, name in self._class_names.items()}
        unknown = [name for name in target_classes if name not in by_name]
        if unknown:
            raise ConfigError(
                f"model.target_classes contains names the model does not know: {unknown}. "
                f"Available classes: {sorted(by_name)}"
            )
        return tuple(by_name[name] for name in target_classes)


__all__ = ["BBox", "Detection", "DetectionResult", "YoloDetector", "resolve_device"]
