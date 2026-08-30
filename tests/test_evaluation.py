"""Detection-quality matching and AP@50 — synthetic boxes, no weights."""

from __future__ import annotations

from smartcity_vision.evaluation.boxes import PredictedBox, TruthBox, box_iou
from smartcity_vision.evaluation.coco import coco_xywh_to_xyxy, select_slice, truths_from_slice
from smartcity_vision.evaluation.metrics import evaluate_detections


def _pred(
    image: str, name: str, bbox: tuple[float, float, float, float], confidence: float
) -> PredictedBox:
    return PredictedBox(image_id=image, class_name=name, bbox=bbox, confidence=confidence)


def _truth(
    image: str,
    name: str,
    bbox: tuple[float, float, float, float],
    is_crowd: bool = False,
) -> TruthBox:
    return TruthBox(image_id=image, class_name=name, bbox=bbox, is_crowd=is_crowd)


def test_iou_of_identical_boxes_is_one() -> None:
    box = (10.0, 10.0, 30.0, 40.0)
    assert box_iou(box, box) == 1.0


def test_iou_of_disjoint_boxes_is_zero() -> None:
    assert box_iou((0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)) == 0.0


def test_iou_of_half_overlap_is_one_third() -> None:
    # 10x10 square overlapping the right half of another 10x10 square: intersection 50,
    # union 150, IoU = 1/3.
    assert box_iou((0.0, 0.0, 10.0, 10.0), (5.0, 0.0, 15.0, 10.0)) == pytest_approx_third()


def pytest_approx_third() -> float:
    return 50.0 / 150.0


def test_perfect_match_is_precision_recall_and_ap_one() -> None:
    report = evaluate_detections(
        [_pred("a", "car", (0, 0, 10, 10), 0.9)],
        [_truth("a", "car", (0, 0, 10, 10))],
        classes=("car",),
    )
    assert report.map50 == 1.0
    assert report.micro_precision == 1.0
    assert report.micro_recall == 1.0
    assert report.per_class[0].true_positives == 1


def test_missed_object_is_a_false_negative_not_an_invented_detection() -> None:
    report = evaluate_detections(
        [],
        [_truth("a", "car", (0, 0, 10, 10))],
        classes=("car",),
    )
    assert report.micro_precision == 0.0
    assert report.micro_recall == 0.0
    assert report.per_class[0].false_negatives == 1
    assert report.per_class[0].ap50 == 0.0


def test_wrong_class_does_not_match() -> None:
    report = evaluate_detections(
        [_pred("a", "person", (0, 0, 10, 10), 0.9)],
        [_truth("a", "car", (0, 0, 10, 10))],
        classes=("car", "person"),
    )
    by_name = {item.class_name: item for item in report.per_class}
    assert by_name["car"].false_negatives == 1
    assert by_name["person"].false_positives == 1
    assert by_name["car"].true_positives == 0


def test_low_iou_is_a_false_positive() -> None:
    report = evaluate_detections(
        [_pred("a", "car", (50, 50, 60, 60), 0.9)],
        [_truth("a", "car", (0, 0, 10, 10))],
        classes=("car",),
        iou_threshold=0.5,
    )
    assert report.per_class[0].true_positives == 0
    assert report.per_class[0].false_positives == 1
    assert report.per_class[0].false_negatives == 1


def test_second_prediction_on_the_same_truth_is_a_false_positive() -> None:
    box = (0.0, 0.0, 10.0, 10.0)
    report = evaluate_detections(
        [_pred("a", "car", box, 0.99), _pred("a", "car", box, 0.80)],
        [_truth("a", "car", box)],
        classes=("car",),
    )
    assert report.per_class[0].true_positives == 1
    assert report.per_class[0].false_positives == 1


def test_crowd_box_absorbs_a_prediction_without_creating_a_false_negative() -> None:
    report = evaluate_detections(
        [_pred("a", "person", (0, 0, 10, 10), 0.9)],
        [
            _truth("a", "person", (0, 0, 10, 10), is_crowd=True),
            _truth("a", "person", (80, 80, 90, 90)),
        ],
        classes=("person",),
    )
    metrics = report.per_class[0]
    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1


def test_ap50_for_half_recall_at_perfect_precision() -> None:
    # One hit, one miss: precision stays 1.0 up to recall 0.5, then 0.
    # 101-point interpolation: 51 samples at 1.0 (recall 0.00..0.50) and 50 at 0.
    report = evaluate_detections(
        [_pred("a", "car", (0, 0, 10, 10), 0.9)],
        [_truth("a", "car", (0, 0, 10, 10)), _truth("b", "car", (0, 0, 10, 10))],
        classes=("car",),
    )
    assert report.per_class[0].recall == 0.5
    assert report.per_class[0].precision == 1.0
    assert round(report.map50, 4) == round(51 / 101, 4)


def test_coco_xywh_converts_to_xyxy() -> None:
    assert coco_xywh_to_xyxy([10.0, 20.0, 5.0, 8.0]) == (10.0, 20.0, 15.0, 28.0)


def test_slice_selection_is_deterministic_and_respects_class_filter() -> None:
    instances = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 10, "height": 10},
            {"id": 2, "file_name": "000000000002.jpg", "width": 10, "height": 10},
            {"id": 3, "file_name": "000000000003.jpg", "width": 10, "height": 10},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 3, "bbox": [0, 0, 4, 4], "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 4, 4], "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 6, "bbox": [0, 0, 4, 4], "iscrowd": 0},
        ],
        "categories": [
            {"id": 1, "name": "person"},
            {"id": 3, "name": "car"},
            {"id": 6, "name": "bus"},
            {"id": 18, "name": "dog"},
        ],
    }
    first = select_slice(instances, classes=("car", "person", "bus"), max_images=2, seed=0)
    second = select_slice(instances, classes=("car", "person", "bus"), max_images=2, seed=0)
    assert [item["id"] for item in first["images"]] == [item["id"] for item in second["images"]]
    assert len(first["images"]) == 2
    names = {item["name"] for item in first["categories"]}
    assert names == {"person", "car", "bus"}
    truths = truths_from_slice(first)
    assert all(box.class_name in names for box in truths)
    assert "dog" not in {box.class_name for box in truths}
