"""Allow ``python -m smartcity_vision`` to run the pipeline."""

from __future__ import annotations

import sys

from smartcity_vision.cli import main

if __name__ == "__main__":
    sys.exit(main())
