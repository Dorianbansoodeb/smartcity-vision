#!/usr/bin/env python3
"""Evaluate YOLOv8n on a labelled COCO val2017 traffic-class slice.

Downloads the official instances file (cached), builds an 80-image slice,
downloads those JPEGs, runs *this repo's* detector, and writes measured
precision / recall / mAP@50. Nothing in the output is estimated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartcity_vision.detection.detector import YoloDetector  # noqa: E402
from smartcity_vision.evaluation.coco import (  # noqa: E402
    DEFAULT_CLASSES,
    download_slice_images,
    ensure_instances,
    load_instances,
    select_slice,
    truths_from_slice,
)
from smartcity_vision.evaluation.runner import evaluate_image_slice  # noqa: E402
from smartcity_vision.exceptions import SmartCityVisionError  # noqa: E402
from smartcity_vision.utils.config import load_config  # noqa: E402
from smartcity_vision.utils.logging import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Download (if needed), run, and persist measured detection quality."""
    parser = argparse.ArgumentParser(description="COCO-slice detection quality evaluation.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/evaluation/cache"))
    parser.add_argument("--image-dir", type=Path, default=Path("data/evaluation/images"))
    parser.add_argument("--slice", type=Path, default=Path("data/evaluation/coco_val80_slice.json"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/evaluation/coco_val80_results.json")
    )
    parser.add_argument("--max-images", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    setup_logging("INFO")
    try:
        instances_path = ensure_instances(args.cache_dir)
        if args.slice.is_file():
            logger.info("Reusing committed slice %s", args.slice)
            slice_json = json.loads(args.slice.read_text(encoding="utf-8"))
        else:
            slice_json = select_slice(
                load_instances(instances_path),
                classes=DEFAULT_CLASSES,
                max_images=args.max_images,
                seed=args.seed,
            )
            args.slice.parent.mkdir(parents=True, exist_ok=True)
            args.slice.write_text(json.dumps(slice_json) + "\n", encoding="utf-8")
            logger.info("Wrote slice manifest to %s", args.slice)

        image_paths = download_slice_images(slice_json, args.image_dir)
        truths = truths_from_slice(slice_json)
        config = load_config(updates={"model": {"device": args.device, "warmup": True}})
        detector = YoloDetector(config.model)
        sample = cv2_size(image_paths[0])
        detector.warmup(sample[0], sample[1])
        report = evaluate_image_slice(detector, image_paths, truths, DEFAULT_CLASSES)

        payload = {
            "dataset": "COCO val2017 traffic-class slice",
            "slice_path": str(args.slice),
            "images": len(image_paths),
            "seed": slice_json.get("info", {}).get("seed", args.seed),
            "weights": str(config.model.weights),
            "device": detector.device,
            "confidence_threshold": config.model.confidence_threshold,
            "iou_threshold": config.model.iou_threshold,
            "image_size": config.model.image_size,
            "classes": list(DEFAULT_CLASSES),
            "metrics": report.as_dict(),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logger.info("Wrote %s", args.output)
        logger.info(
            "mAP@50=%.4f  P=%.4f  R=%.4f  images=%d",
            report.map50,
            report.micro_precision,
            report.micro_recall,
            report.images,
        )
        for row in report.per_class:
            logger.info(
                "  %-12s AP50=%.3f P=%.3f R=%.3f GT=%d",
                row.class_name,
                row.ap50,
                row.precision,
                row.recall,
                row.ground_truths,
            )
    except SmartCityVisionError as exc:
        logger.error("%s", exc)
        return 1
    return 0


def cv2_size(path: Path) -> tuple[int, int]:
    """Return ``(width, height)`` for an image path without keeping the buffer."""
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise SmartCityVisionError(f"Could not read {path}")
    height, width = image.shape[:2]
    return width, height


if __name__ == "__main__":
    raise SystemExit(main())
