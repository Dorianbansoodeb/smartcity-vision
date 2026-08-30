#!/usr/bin/env python3
"""Champion/challenger: run the same clip through two weight files.

Precision/recall/mAP are not computed (the clip has no labels). The report
compares measured latency and detection counts only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartcity_vision.cli import run  # noqa: E402
from smartcity_vision.experiments import write_challenger_report  # noqa: E402
from smartcity_vision.utils.config import load_config  # noqa: E402
from smartcity_vision.utils.logging import setup_logging  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run two configured models and write a comparison report."""
    parser = argparse.ArgumentParser(description="Champion/challenger comparison.")
    parser.add_argument("--champion", default="models/yolov8n.pt")
    parser.add_argument("--challenger", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("data/output/champion_challenger.json"))
    args = parser.parse_args(argv)

    setup_logging("INFO")
    results = []
    for label, weights in (("champion", args.champion), ("challenger", args.challenger)):
        updates = {
            "model": {"weights": weights},
            "output": {
                "directory": Path("data/output") / label,
                "write_annotated_video": False,
            },
            "video": {"max_frames": args.max_frames},
        }
        if args.input:
            updates["video"]["source"] = args.input
        stats = run(load_config(updates=updates))
        payload = stats.as_dict()
        payload["label"] = label
        payload["weights"] = weights
        results.append(payload)

    write_challenger_report(results[0], results[1], args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
