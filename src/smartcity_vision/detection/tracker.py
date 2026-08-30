"""Multi-object tracking.

A tracker is a detector that also assigns identities, so :class:`YoloTracker`
subclasses :class:`~smartcity_vision.detection.detector.YoloDetector` and
overrides only the model call. Everything downstream keeps consuming
``DetectionResult``; the difference is that detections now carry a ``track_id``.

Tracking state lives inside the Ultralytics model instance and depends on being
called once per frame in order, which is why the model is loaded once and the
same object is reused for a whole run.
"""

from __future__ import annotations

import numpy as np

from smartcity_vision.detection.detector import YoloDetector
from smartcity_vision.utils.config import ModelConfig, TrackingConfig
from smartcity_vision.utils.logging import get_logger

logger = get_logger(__name__)


class YoloTracker(YoloDetector):
    """Runs YOLOv8 tracking, returning detections with persistent track IDs."""

    def __init__(self, model_config: ModelConfig, tracking_config: TrackingConfig) -> None:
        """Load weights and configure the tracker.

        Args:
            model_config: Validated model configuration.
            tracking_config: Validated tracking configuration.
        """
        super().__init__(model_config)
        self._tracking_config = tracking_config
        logger.info("Tracking enabled using %s", tracking_config.tracker)

    @property
    def tracker_name(self) -> str:
        """Name of the Ultralytics tracker configuration in use."""
        return self._tracking_config.tracker

    def _run(self, image: np.ndarray) -> list:
        """Track through this frame, carrying identities over from the last one.

        ``persist=True`` is what stops the tracker being reinitialised on every
        call; without it each frame would start fresh and every object would be
        assigned a new ID.
        """
        return self._model.track(
            source=image,
            persist=True,
            tracker=self._tracking_config.tracker,
            **self._inference_kwargs(),
        )


__all__ = ["YoloTracker"]
