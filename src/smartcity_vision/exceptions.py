"""Exception hierarchy for SmartCity Vision.

A single hierarchy keeps failure handling predictable: callers can catch
:class:`SmartCityVisionError` to trap anything raised deliberately by this
package, while still distinguishing configuration problems from runtime
video/model failures.
"""

from __future__ import annotations


class SmartCityVisionError(Exception):
    """Base class for every error raised deliberately by this package."""


class ConfigError(SmartCityVisionError):
    """Raised when configuration is missing, malformed, or semantically invalid."""


class VideoSourceError(SmartCityVisionError):
    """Raised when a video source cannot be opened, read, or written."""


class DetectionError(SmartCityVisionError):
    """Raised when model weights cannot be loaded or inference fails."""


__all__ = [
    "ConfigError",
    "DetectionError",
    "SmartCityVisionError",
    "VideoSourceError",
]
