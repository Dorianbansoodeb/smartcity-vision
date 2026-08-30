"""Bounding-box types and intersection-over-union."""

from __future__ import annotations

from dataclasses import dataclass

BBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PredictedBox:
    """One model prediction on one image."""

    image_id: str
    class_name: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True, slots=True)
class TruthBox:
    """One ground-truth object on one image.

    Crowd boxes (COCO ``iscrowd=1``) are ignore regions: a prediction may match
    them so it is not counted as a false positive, but they never create a
    false negative.
    """

    image_id: str
    class_name: str
    bbox: BBox
    is_crowd: bool = False


def box_iou(first: BBox, second: BBox) -> float:
    """Intersection-over-union of two ``(x1, y1, x2, y2)`` boxes.

    Degenerate boxes (no area) return ``0.0`` rather than dividing by zero.
    """
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0.0:
        return 0.0
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


__all__ = ["BBox", "PredictedBox", "TruthBox", "box_iou"]
