#!/usr/bin/env python3
"""Analyse a live webcam feed.

Example:
    python scripts/run_webcam.py --display
    python scripts/run_webcam.py --input 1   # second camera
"""

from __future__ import annotations

import sys
from pathlib import Path

# Support running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smartcity_vision.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(default_source="0"))
