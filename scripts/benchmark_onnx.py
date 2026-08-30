#!/usr/bin/env python3
"""Compare PyTorch vs ONNX Runtime inference latency on the same clip.

Numbers written to data/output/benchmark_results.json come from this process,
not from estimates. Export happens once and is cached next to the .pt weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartcity_vision.detection.detector import YoloDetector, resolve_device  # noqa: E402
from smartcity_vision.utils.config import ModelConfig, load_config  # noqa: E402
from smartcity_vision.utils.logging import get_logger, setup_logging  # noqa: E402
from smartcity_vision.video.source import create_video_source  # noqa: E402

logger = get_logger(__name__)


def _measure(detector: YoloDetector, frames: list, warmup: int) -> dict[str, float]:
    detector.warmup(frames[0].image.shape[1], frames[0].image.shape[0])
    samples: list[float] = []
    for index, frame in enumerate(frames):
        result = detector.detect(frame)
        if index >= warmup:
            samples.append(result.inference_ms)
    samples.sort()
    return {
        "frames": float(len(samples)),
        "avg_ms": sum(samples) / len(samples),
        "p95_ms": samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))],
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write measured results."""
    parser = argparse.ArgumentParser(description="PyTorch vs ONNX Runtime latency benchmark.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cpu", help="Keep CPU for a fair ORT comparison")
    parser.add_argument("--output", type=Path, default=Path("data/output/benchmark_results.json"))
    args = parser.parse_args(argv)

    setup_logging("INFO")
    config = load_config()
    source_path = args.input or config.video.source
    frames = []
    with create_video_source(source_path) as source:
        for frame in source:
            frames.append(frame)
            if len(frames) >= args.frames + args.warmup:
                break
    if len(frames) <= args.warmup:
        logger.error("Not enough frames in %s", source_path)
        return 1

    pt_config = ModelConfig(
        weights=config.model.weights,
        device=resolve_device(args.device),
        confidence_threshold=config.model.confidence_threshold,
        iou_threshold=config.model.iou_threshold,
        image_size=config.model.image_size,
        warmup=False,
    )
    logger.info("Measuring PyTorch on %d frames after %d warmup", args.frames, args.warmup)
    pytorch = _measure(YoloDetector(pt_config), frames, args.warmup)

    onnx_path = Path(config.model.weights).with_suffix(".onnx")
    if not onnx_path.is_file():
        logger.info("Exporting %s -> %s", config.model.weights, onnx_path)
        from ultralytics import YOLO

        YOLO(str(config.model.weights)).export(format="onnx", imgsz=config.model.image_size)
        exported = Path(config.model.weights).with_suffix(".onnx")
        if exported != onnx_path and exported.is_file():
            exported.replace(onnx_path)
    logger.info("Measuring ONNX Runtime on the same frames")
    onnx = _measure(
        YoloDetector(pt_config.model_copy(update={"weights": onnx_path})), frames, args.warmup
    )

    payload = {
        "source": source_path,
        "device": args.device,
        "image_size": config.model.image_size,
        "measured_frames": args.frames,
        "warmup_frames": args.warmup,
        "pytorch": {key: round(value, 3) for key, value in pytorch.items()},
        "onnxruntime": {key: round(value, 3) for key, value in onnx.items()},
        "speedup_avg": round(pytorch["avg_ms"] / onnx["avg_ms"], 3) if onnx["avg_ms"] else None,
        "note": "Values measured by this process. speedup_avg > 1 means ONNX was faster.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", args.output)
    logger.info(
        "PyTorch avg %.2f ms | ONNX avg %.2f ms | speedup %.2fx",
        pytorch["avg_ms"],
        onnx["avg_ms"],
        payload["speedup_avg"] or 0.0,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
