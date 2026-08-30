"""Face and licence-plate anonymisation.

Public camera footage of people is exactly the kind of data GDPR/PIPEDA-style
rules require you to handle carefully. This step runs *before* any frame is
written to disk when ``privacy.enabled`` is true (the default).

Detection uses OpenCV Haar cascades shipped with the library. They are not a
production-grade PII detector — they miss profile faces, small plates, and
unusual angles — but they make the compliance control *real* rather than a
comment in a README, and the test suite asserts that pixels inside a detected
region are actually altered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from smartcity_vision.detection.detector import Detection
from smartcity_vision.utils.config import PrivacyConfig
from smartcity_vision.utils.logging import get_logger

_VEHICLE_CLASSES = frozenset({"car", "truck", "bus", "motorcycle"})

logger = get_logger(__name__)

Region = tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True, slots=True)
class AnonymizationStats:
    """How many regions were redacted on one frame."""

    faces: int
    plates: int

    @property
    def total(self) -> int:
        """Faces plus plates."""
        return self.faces + self.plates


class FrameAnonymizer:
    """Blurs or pixelates faces and licence plates on a frame."""

    def __init__(self, config: PrivacyConfig) -> None:
        """Load cascade classifiers.

        Args:
            config: Validated privacy configuration.
        """
        self._config = config
        self._face = _load_cascade("haarcascade_frontalface_default.xml")
        self._plate = _load_cascade("haarcascade_russian_plate_number.xml")
        if config.enabled:
            logger.info(
                "Privacy anonymisation ON (method=%s, strength=%d)",
                config.method,
                config.blur_strength,
            )

    @property
    def enabled(self) -> bool:
        """Whether frames will be redacted before they are persisted."""
        return self._config.enabled

    def anonymize(
        self,
        image: np.ndarray,
        detections: list[Detection] | None = None,
    ) -> tuple[np.ndarray, AnonymizationStats]:
        """Return a redacted copy of ``image`` and the region counts.

        Haar cascades are used when OpenCV ships them. On wheels that do not
        (OpenCV 5 on this machine does not), fallback regions are derived from
        detections: the top of a person box (face) and the bottom of a vehicle
        box (plate). The input is never mutated.

        Args:
            image: BGR frame.
            detections: Optional detections used for the geometry fallback.

        Returns:
            The redacted copy and how many regions were altered.
        """
        if not self._config.enabled:
            return image, AnonymizationStats(faces=0, plates=0)

        output = image.copy()
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        faces = _detect(self._face, gray)
        plates = _detect(self._plate, gray)
        if detections:
            faces.extend(_face_fallback(detections))
            plates.extend(_plate_fallback(detections))
        for region in faces:
            self._redact(output, region)
        for region in plates:
            self._redact(output, region)
        return output, AnonymizationStats(faces=len(faces), plates=len(plates))

    def redact_regions(self, image: np.ndarray, regions: list[Region]) -> np.ndarray:
        """Redact explicit regions. Used by tests that inject a known box."""
        output = image.copy()
        for region in regions:
            self._redact(output, region)
        return output

    def _redact(self, image: np.ndarray, region: Region) -> None:
        """Blur or pixelate ``region`` in place."""
        x, y, width, height = _clamp_region(region, image.shape)
        if width <= 0 or height <= 0:
            return
        roi = image[y : y + height, x : x + width]
        if self._config.method == "pixelate":
            image[y : y + height, x : x + width] = _pixelate(roi, self._config.pixel_size)
            return
        strength = self._config.blur_strength
        if strength % 2 == 0:
            strength += 1
        image[y : y + height, x : x + width] = cv2.GaussianBlur(roi, (strength, strength), 0)


def _load_cascade(filename: str) -> Any:
    """Load a Haar cascade shipped with OpenCV, or ``None`` if it is absent."""
    cascade_dir = Path(cv2.data.haarcascades)  # type: ignore[attr-defined]
    path = cascade_dir / filename
    if not path.is_file():
        logger.warning("Haar cascade %s is not installed; that detector is disabled", filename)
        return None
    classifier = cv2.CascadeClassifier(str(path))  # type: ignore[attr-defined]
    if classifier.empty():
        logger.warning("Could not load Haar cascade %s", path)
        return None
    return classifier


def _detect(classifier: Any, gray: np.ndarray) -> list[Region]:
    """Run a cascade, returning an empty list when the classifier is missing."""
    if classifier is None:
        return []
    detections = classifier.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20)
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in detections]


def _pixelate(roi: np.ndarray, pixel_size: int) -> np.ndarray:
    """Downscale then upscale ``roi`` so it becomes a mosaic."""
    height, width = roi.shape[:2]
    small_w = max(1, width // pixel_size)
    small_h = max(1, height // pixel_size)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _face_fallback(detections: list[Detection]) -> list[Region]:
    """Top 35% of each person box — a conservative face stand-in."""
    regions: list[Region] = []
    for detection in detections:
        if detection.class_name != "person":
            continue
        x1, y1, x2, y2 = detection.int_bbox()
        height = max(1, y2 - y1)
        regions.append((x1, y1, max(1, x2 - x1), max(1, int(height * 0.35))))
    return regions


def _plate_fallback(detections: list[Detection]) -> list[Region]:
    """Bottom 30% of each vehicle box — a conservative plate stand-in."""
    regions: list[Region] = []
    for detection in detections:
        if detection.class_name not in _VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = detection.int_bbox()
        height = max(1, y2 - y1)
        plate_h = max(1, int(height * 0.30))
        regions.append((x1, y2 - plate_h, max(1, x2 - x1), plate_h))
    return regions


def _clamp_region(region: Region, shape: tuple[int, ...]) -> Region:
    """Keep a region inside the frame."""
    height, width = shape[:2]
    x, y, w, h = region
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    return (x, y, w, h)


__all__ = ["AnonymizationStats", "FrameAnonymizer", "Region"]
