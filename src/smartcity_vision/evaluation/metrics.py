"""Precision, recall, and COCO-style AP@50.

Per-class AP uses the 101-point interpolated precision-recall curve (the COCO
``mAP@0.50`` definition). Operating-point precision and recall use the
detections the model already emitted — they are not swept over confidence.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from smartcity_vision.evaluation.boxes import PredictedBox, TruthBox, box_iou

_AP_POINTS = 101


@dataclass(frozen=True, slots=True)
class ClassMetrics:
    """Measured quality for one class on one evaluation set."""

    class_name: str
    ground_truths: int
    predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    ap50: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable record."""
        return {
            "class_name": self.class_name,
            "ground_truths": self.ground_truths,
            "predictions": self.predictions,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "ap50": round(self.ap50, 4),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """Aggregate detection quality across classes."""

    iou_threshold: float
    images: int
    map50: float
    micro_precision: float
    micro_recall: float
    per_class: tuple[ClassMetrics, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "iou_threshold": self.iou_threshold,
            "images": self.images,
            "map50": round(self.map50, 4),
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "per_class": [item.as_dict() for item in self.per_class],
            "notes": list(self.notes),
        }


def evaluate_detections(
    predictions: list[PredictedBox],
    truths: list[TruthBox],
    classes: tuple[str, ...],
    iou_threshold: float = 0.5,
) -> EvaluationReport:
    """Score predictions against ground truth at a single IoU threshold.

    Args:
        predictions: Model boxes. Already confidence-filtered by the detector.
        truths: Labelled boxes. Crowd boxes are ignore regions only.
        classes: Classes to score. A class with no ground truth is omitted
            from the mAP average rather than reported as AP = 0.
        iou_threshold: Minimum IoU to count a match. ``0.5`` is mAP@50.

    Returns:
        Per-class and aggregate metrics computed from these two lists.
    """
    image_ids = {box.image_id for box in predictions} | {box.image_id for box in truths}
    scored: list[ClassMetrics] = []
    for name in classes:
        metrics = _score_class(name, predictions, truths, iou_threshold)
        if metrics.ground_truths == 0 and metrics.predictions == 0:
            continue
        scored.append(metrics)
    per_class = tuple(scored)
    with_ground_truth = tuple(item for item in per_class if item.ground_truths > 0)
    true_positives = sum(item.true_positives for item in per_class)
    false_positives = sum(item.false_positives for item in per_class)
    false_negatives = sum(item.false_negatives for item in per_class)
    predicted = true_positives + false_positives
    labelled = true_positives + false_negatives
    map50 = (
        0.0
        if not with_ground_truth
        else sum(item.ap50 for item in with_ground_truth) / len(with_ground_truth)
    )
    return EvaluationReport(
        iou_threshold=iou_threshold,
        images=len(image_ids),
        map50=map50,
        micro_precision=_ratio(true_positives, predicted),
        micro_recall=_ratio(true_positives, labelled),
        per_class=per_class,
        notes=(
            "AP@50 is the 101-point interpolated curve (COCO mAP@0.50).",
            "Precision and recall are at the detector's operating point, not swept.",
            "Crowd boxes are ignore regions and do not create false negatives.",
            "mAP averages only classes that have at least one non-crowd ground truth.",
        ),
    )


def _score_class(
    class_name: str,
    predictions: list[PredictedBox],
    truths: list[TruthBox],
    iou_threshold: float,
) -> ClassMetrics:
    class_preds = sorted(
        (item for item in predictions if item.class_name == class_name),
        key=lambda item: item.confidence,
        reverse=True,
    )
    class_truths = [item for item in truths if item.class_name == class_name and not item.is_crowd]
    crowd = [item for item in truths if item.class_name == class_name and item.is_crowd]
    matched: set[int] = set()
    crowd_by_image: dict[str, list[TruthBox]] = defaultdict(list)
    truth_by_image: dict[str, list[tuple[int, TruthBox]]] = defaultdict(list)
    for index, truth in enumerate(class_truths):
        truth_by_image[truth.image_id].append((index, truth))
    for item in crowd:
        crowd_by_image[item.image_id].append(item)

    ranks: list[bool] = []
    true_positives = 0
    false_positives = 0
    for prediction in class_preds:
        match_index = _best_unmatched(
            prediction, truth_by_image.get(prediction.image_id, []), matched, iou_threshold
        )
        if match_index is not None:
            matched.add(match_index)
            ranks.append(True)
            true_positives += 1
            continue
        if _hits_crowd(prediction, crowd_by_image.get(prediction.image_id, []), iou_threshold):
            continue
        ranks.append(False)
        false_positives += 1

    ground_truths = len(class_truths)
    false_negatives = ground_truths - true_positives
    return ClassMetrics(
        class_name=class_name,
        ground_truths=ground_truths,
        predictions=len(class_preds),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=_ratio(true_positives, true_positives + false_positives),
        recall=_ratio(true_positives, ground_truths),
        ap50=_average_precision(ranks, ground_truths),
    )


def _best_unmatched(
    prediction: PredictedBox,
    candidates: list[tuple[int, TruthBox]] | tuple[tuple[int, TruthBox], ...],
    matched: set[int],
    iou_threshold: float,
) -> int | None:
    best_index: int | None = None
    best_iou = iou_threshold
    for index, truth in candidates:
        if index in matched:
            continue
        iou = box_iou(prediction.bbox, truth.bbox)
        if iou >= best_iou:
            best_iou = iou
            best_index = index
    return best_index


def _hits_crowd(prediction: PredictedBox, crowds: list[TruthBox], iou_threshold: float) -> bool:
    return any(box_iou(prediction.bbox, crowd.bbox) >= iou_threshold for crowd in crowds)


def _average_precision(ranked_true_positives: list[bool], ground_truths: int) -> float:
    """101-point interpolated AP.

    Args:
        ranked_true_positives: Matches in decreasing-confidence order.
        ground_truths: Positive count for this class. Zero yields AP = 0.
    """
    if ground_truths <= 0:
        return 0.0
    true_so_far = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for index, is_tp in enumerate(ranked_true_positives, start=1):
        if is_tp:
            true_so_far += 1
        recalls.append(true_so_far / ground_truths)
        precisions.append(true_so_far / index)
    if not recalls:
        return 0.0
    return _interpolate_101(recalls, precisions)


def _interpolate_101(recalls: list[float], precisions: list[float]) -> float:
    """COCO 101-point interpolation: mean precision at recall 0.00, 0.01, …, 1.00."""
    sampled = 0.0
    for step in range(_AP_POINTS):
        threshold = step / (_AP_POINTS - 1)
        above = [precisions[index] for index, recall in enumerate(recalls) if recall >= threshold]
        sampled += max(above) if above else 0.0
    return sampled / _AP_POINTS


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


__all__ = ["ClassMetrics", "EvaluationReport", "evaluate_detections"]
