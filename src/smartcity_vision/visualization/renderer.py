"""Overlay rendering with OpenCV.

Colours are derived deterministically from the class name, so the same class is
always drawn in the same colour across runs and videos without maintaining a
palette by hand. Everything that affects appearance comes from
:class:`VisualizationConfig`; nothing is hardcoded per video.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import cv2
import numpy as np

from smartcity_vision.detection.detector import Detection
from smartcity_vision.utils.config import VisualizationConfig

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_COLOR = (255, 255, 255)
_HUD_BACKGROUND = (0, 0, 0)
_HUD_ALPHA = 0.55
_HUD_MARGIN = 10
_HUD_PADDING = 8

Color = tuple[int, int, int]


def color_for_class(class_name: str) -> Color:
    """Return a stable, saturated BGR colour for ``class_name``.

    The class name is hashed to a hue, so colours are consistent across runs and
    distinct classes rarely collide.
    """
    digest = hashlib.sha256(class_name.encode("utf-8")).digest()
    hue = digest[0] % 180  # OpenCV hue range is 0-179
    hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


class FrameRenderer:
    """Draws detections and a status panel onto frames."""

    def __init__(self, config: VisualizationConfig) -> None:
        """Initialise the renderer.

        Args:
            config: Validated visualization configuration.
        """
        self._config = config
        self._color_cache: dict[str, Color] = {
            name.lower(): tuple(colour) for name, colour in config.class_colors.items()
        }

    def color_of(self, class_name: str) -> Color:
        """Return the colour used for ``class_name``, honouring config overrides."""
        key = class_name.lower()
        if key not in self._color_cache:
            self._color_cache[key] = color_for_class(key)
        return self._color_cache[key]

    def draw_detections(
        self,
        image: np.ndarray,
        detections: Sequence[Detection],
    ) -> np.ndarray:
        """Draw bounding boxes and labels in place.

        Args:
            image: BGR frame, modified in place.
            detections: Detections to draw.

        Returns:
            The same array, for call chaining.
        """
        for detection in detections:
            colour = self.color_of(detection.class_name)
            x1, y1, x2, y2 = detection.int_bbox()
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, self._config.box_thickness)
            if self._config.show_labels:
                self._draw_label(image, self._label_for(detection), (x1, y1), colour)
        return image

    def draw_hud(self, image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
        """Draw a translucent status panel in the configured corner.

        Args:
            image: BGR frame, modified in place.
            lines: Text lines to display, e.g. ``["FPS: 12.4", "Objects: 7"]``.

        Returns:
            The same array, for call chaining.
        """
        if not self._config.show_hud or not lines:
            return image

        scale = self._config.font_scale * 1.2
        thickness = max(1, self._config.box_thickness - 1)
        sizes = [cv2.getTextSize(line, _FONT, scale, thickness)[0] for line in lines]
        line_height = max(height for _, height in sizes)
        line_gap = round(line_height * 0.7)

        panel_width = max(width for width, _ in sizes) + 2 * _HUD_PADDING
        panel_height = len(lines) * line_height + (len(lines) - 1) * line_gap + 2 * _HUD_PADDING
        origin_x, origin_y = self._hud_origin(image.shape[:2], panel_width, panel_height)

        overlay = image.copy()
        cv2.rectangle(
            overlay,
            (origin_x, origin_y),
            (origin_x + panel_width, origin_y + panel_height),
            _HUD_BACKGROUND,
            cv2.FILLED,
        )
        cv2.addWeighted(overlay, _HUD_ALPHA, image, 1.0 - _HUD_ALPHA, 0, dst=image)

        baseline = origin_y + _HUD_PADDING + line_height
        for line in lines:
            cv2.putText(
                image,
                line,
                (origin_x + _HUD_PADDING, baseline),
                _FONT,
                scale,
                _TEXT_COLOR,
                thickness,
                cv2.LINE_AA,
            )
            baseline += line_height + line_gap
        return image

    def _hud_origin(
        self,
        frame_shape: tuple[int, int],
        panel_width: int,
        panel_height: int,
    ) -> tuple[int, int]:
        """Return the panel's top-left pixel for the configured corner."""
        height, width = frame_shape
        right = max(_HUD_MARGIN, width - panel_width - _HUD_MARGIN)
        bottom = max(_HUD_MARGIN, height - panel_height - _HUD_MARGIN)
        corners = {
            "top-left": (_HUD_MARGIN, _HUD_MARGIN),
            "top-right": (right, _HUD_MARGIN),
            "bottom-left": (_HUD_MARGIN, bottom),
            "bottom-right": (right, bottom),
        }
        return corners[self._config.hud_position]

    def _label_for(self, detection: Detection) -> str:
        """Build the box label text: class, optional track ID, optional confidence."""
        parts = [detection.class_name]
        if self._config.show_track_ids and detection.track_id is not None:
            parts.append(f"#{detection.track_id}")
        if self._config.show_confidence:
            parts.append(f"{detection.confidence:.2f}")
        return " ".join(parts)

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        anchor: tuple[int, int],
        colour: Color,
    ) -> None:
        """Draw ``text`` on a filled tag above ``anchor``, clamped to the frame."""
        scale = self._config.font_scale
        thickness = max(1, self._config.box_thickness - 1)
        (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, scale, thickness)

        x = max(0, min(anchor[0], image.shape[1] - text_width - 4))
        tag_height = text_height + baseline + 4
        # Prefer sitting above the box; drop inside it when there is no room.
        top = anchor[1] - tag_height if anchor[1] - tag_height >= 0 else anchor[1]

        cv2.rectangle(
            image,
            (x, top),
            (x + text_width + 4, top + tag_height),
            colour,
            cv2.FILLED,
        )
        cv2.putText(
            image,
            text,
            (x + 2, top + text_height + 2),
            _FONT,
            scale,
            _TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )


__all__ = ["Color", "FrameRenderer", "color_for_class"]
