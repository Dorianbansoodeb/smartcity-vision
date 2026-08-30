"""Run the project detector against a labelled image slice."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2

from smartcity_vision.detection.detector import YoloDetector
from smartcity_vision.evaluation.boxes import PredictedBox, TruthBox
from smartcity_vision.evaluation.coco import image_id_from_path
from smartcity_vision.evaluation.metrics import EvaluationReport, evaluate_detections
from smartcity_vision.exceptions import DetectionError
from smartcity_vision.video.source import Frame


def detect_image(detector: YoloDetector, path: Path) -> list[PredictedBox]:
    """Run the detector on one image file."""
    image = cv2.imread(str(path))
    if image is None:
        raise DetectionError(f"Could not read image {path}")
    result = detector.detect(Frame(index=0, timestamp=0.0, image=image))
    image_id = image_id_from_path(path)
    return [
        PredictedBox(
            image_id=image_id,
            class_name=detection.class_name,
            bbox=detection.bbox,
            confidence=detection.confidence,
        )
        for detection in result.detections
    ]


def evaluate_image_slice(
    detector: YoloDetector,
    image_paths: Sequence[Path],
    truths: list[TruthBox],
    classes: tuple[str, ...],
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    """Detect every image and score the collected boxes."""
    predictions: list[PredictedBox] = []
    for path in image_paths:
        predictions.extend(detect_image(detector, path))
    return evaluate_detections(predictions, truths, classes, iou_threshold=iou_threshold)


__all__ = ["detect_image", "evaluate_image_slice"]
