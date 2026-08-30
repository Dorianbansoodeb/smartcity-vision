"""Detection-quality evaluation.

The parking-lot clip is unlabelled, so precision / recall / mAP are computed on
a labelled COCO val2017 slice instead. Matching and AP live here so the numbers
come from this code, not from a hidden ``model.val()`` call.
"""

from smartcity_vision.evaluation.boxes import PredictedBox, TruthBox, box_iou
from smartcity_vision.evaluation.metrics import (
    ClassMetrics,
    EvaluationReport,
    evaluate_detections,
)

__all__ = [
    "ClassMetrics",
    "EvaluationReport",
    "PredictedBox",
    "TruthBox",
    "box_iou",
    "evaluate_detections",
]
