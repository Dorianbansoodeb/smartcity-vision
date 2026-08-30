"""Behaviour of the overlay renderer."""

from __future__ import annotations

import numpy as np
import pytest

from smartcity_vision.detection.detector import Detection
from smartcity_vision.utils.config import VisualizationConfig
from smartcity_vision.visualization.renderer import FrameRenderer, color_for_class


def blank(height: int = 200, width: int = 300) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def car(bbox: tuple[float, float, float, float] = (50.0, 60.0, 150.0, 140.0)) -> Detection:
    return Detection(class_id=2, class_name="car", confidence=0.87, bbox=bbox)


def test_class_colours_are_stable_and_class_specific() -> None:
    assert color_for_class("car") == color_for_class("car")
    assert color_for_class("car") != color_for_class("truck")


def test_configured_colour_override_is_used_for_the_box() -> None:
    override = (10, 200, 30)
    renderer = FrameRenderer(VisualizationConfig(class_colors={"car": override}))
    image = blank()

    renderer.draw_detections(image, [car()])

    assert renderer.color_of("car") == override
    # Top edge of the box, away from the label tag, must carry the override colour.
    assert tuple(int(channel) for channel in image[60, 100]) == override


def test_box_is_drawn_on_the_outline_and_not_filled() -> None:
    renderer = FrameRenderer(VisualizationConfig(show_labels=False, show_hud=False))
    image = blank()

    renderer.draw_detections(image, [car()])

    assert image[60, 100].any(), "expected the top edge of the box to be drawn"
    assert image[140, 100].any(), "expected the bottom edge of the box to be drawn"
    assert not image[100, 100].any(), "box interior must stay untouched"
    assert not image[10, 10].any(), "pixels outside the box must stay untouched"


def test_label_tag_is_drawn_above_the_box_when_there_is_room() -> None:
    renderer = FrameRenderer(VisualizationConfig(show_labels=True, show_hud=False))
    with_label = blank()
    without_label = blank()

    renderer.draw_detections(with_label, [car()])
    FrameRenderer(VisualizationConfig(show_labels=False, show_hud=False)).draw_detections(
        without_label, [car()]
    )

    above_box = np.s_[40:59, 50:150]
    assert with_label[above_box].any(), "label tag should occupy space above the box"
    assert not without_label[above_box].any()


def test_label_stays_inside_the_frame_for_a_box_at_the_top_left_corner() -> None:
    renderer = FrameRenderer(VisualizationConfig())
    image = blank()

    # A box touching y=0 has no room above it; drawing must not raise or wrap.
    renderer.draw_detections(image, [car(bbox=(0.0, 0.0, 40.0, 30.0))])

    assert image[0:30, 0:40].any()


def test_track_id_appears_in_the_label_only_when_present_and_enabled() -> None:
    tracked = Detection(
        class_id=2, class_name="car", confidence=0.87, bbox=(0, 0, 9, 9), track_id=12
    )
    untracked = car()

    with_ids = FrameRenderer(VisualizationConfig(show_track_ids=True))
    without_ids = FrameRenderer(VisualizationConfig(show_track_ids=False))

    assert with_ids._label_for(tracked) == "car #12 0.87"  # noqa: SLF001
    assert without_ids._label_for(tracked) == "car 0.87"  # noqa: SLF001
    # A detector-only run has no identities, so no stray "#None" in the label.
    assert with_ids._label_for(untracked) == "car 0.87"  # noqa: SLF001


def test_hud_draws_text_in_the_top_left_and_can_be_disabled() -> None:
    enabled = blank()
    disabled = blank()

    FrameRenderer(VisualizationConfig(show_hud=True)).draw_hud(enabled, ["FPS 12.3", "Objects 4"])
    FrameRenderer(VisualizationConfig(show_hud=False)).draw_hud(disabled, ["FPS 12.3"])

    assert enabled[10:60, 10:120].any()
    assert not disabled.any()


@pytest.mark.parametrize(
    ("position", "occupied", "empty"),
    [
        ("top-left", np.s_[10:40, 10:60], np.s_[160:190, 240:290]),
        ("top-right", np.s_[10:40, 240:290], np.s_[160:190, 10:60]),
        ("bottom-left", np.s_[160:190, 10:60], np.s_[10:40, 240:290]),
        ("bottom-right", np.s_[160:190, 240:290], np.s_[10:40, 10:60]),
    ],
)
def test_hud_is_drawn_in_the_configured_corner(
    position: str, occupied: tuple, empty: tuple
) -> None:
    renderer = FrameRenderer(VisualizationConfig(hud_position=position))  # type: ignore[arg-type]
    image = np.full((200, 300, 3), 255, dtype=np.uint8)

    renderer.draw_hud(image, ["FPS 12.3", "Objects 4"])

    # The translucent panel darkens its corner; the opposite corner stays white.
    assert image[occupied].mean() < 250
    assert image[empty].mean() == 255


def test_hud_larger_than_the_frame_is_clamped_inside_it() -> None:
    renderer = FrameRenderer(VisualizationConfig(hud_position="bottom-right"))
    image = np.zeros((40, 60, 3), dtype=np.uint8)

    renderer.draw_hud(image, ["a very long status line that cannot possibly fit"])

    assert image.any(), "the panel must still be drawn rather than clipped away entirely"


def test_hud_with_no_lines_leaves_the_frame_unchanged() -> None:
    image = blank()

    FrameRenderer(VisualizationConfig()).draw_hud(image, [])

    assert not image.any()


def test_detection_geometry_helpers() -> None:
    detection = car(bbox=(10.0, 20.0, 30.0, 60.0))

    assert detection.width == pytest.approx(20.0)
    assert detection.height == pytest.approx(40.0)
    assert detection.area == pytest.approx(800.0)
    assert detection.center == pytest.approx((20.0, 40.0))
    assert detection.int_bbox() == (10, 20, 30, 60)
