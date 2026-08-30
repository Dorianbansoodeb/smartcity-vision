"""Logging setup.

The package never prints; every runtime message goes through the standard
:mod:`logging` module so that verbosity and destinations stay configurable and
so that log output can later be shipped to a real aggregator.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries are chatty at INFO; keep them at WARNING unless debugging.
_NOISY_LOGGERS = ("matplotlib", "PIL", "urllib3")


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure root logging for a CLI run.

    Safe to call more than once: existing handlers installed by this function
    are replaced rather than duplicated.

    Args:
        level: Logging level name, e.g. ``"DEBUG"`` or ``"INFO"``.
        log_file: Optional file that receives the same records as stderr. Parent
            directories are created if needed.
    """
    numeric_level = logging.getLevelNamesMapping().get(level.upper())
    if numeric_level is None:
        raise ValueError(f"Unknown logging level: {level!r}")

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(stream=sys.stderr)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(numeric_level)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(max(numeric_level, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    """Return the module-scoped logger for ``name``."""
    return logging.getLogger(name)


__all__ = ["get_logger", "setup_logging"]
