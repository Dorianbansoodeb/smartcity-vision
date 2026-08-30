"""Behaviour of the shared 2-D geometry helpers."""

from __future__ import annotations

import pytest

from smartcity_vision.utils.geometry import (
    euclidean_distance,
    point_in_polygon,
    polyline_length,
    segments_intersect,
    side_of_line,
)

SQUARE = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))


def test_side_of_line_is_left_right_or_on() -> None:
    start, end = (0.0, 0.0), (10.0, 0.0)

    assert side_of_line((5.0, 4.0), start, end) == 1
    assert side_of_line((5.0, -3.0), start, end) == -1
    assert side_of_line((5.0, 0.0), start, end) == 0


def test_side_of_line_flips_when_the_segment_is_reversed() -> None:
    point = (5.0, 4.0)

    assert side_of_line(point, (0.0, 0.0), (10.0, 0.0)) == -side_of_line(
        point, (10.0, 0.0), (0.0, 0.0)
    )


def test_crossing_segments_are_reported() -> None:
    assert segments_intersect(((0.0, 0.0), (10.0, 10.0)), ((0.0, 10.0), (10.0, 0.0)))
    assert not segments_intersect(((0.0, 0.0), (4.0, 0.0)), ((6.0, 0.0), (10.0, 0.0)))


def test_a_trajectory_that_lands_on_the_line_still_intersects() -> None:
    line = ((0.0, 5.0), (10.0, 5.0))
    step = ((5.0, 0.0), (5.0, 5.0))

    assert segments_intersect(step, line)


def test_a_path_that_misses_the_end_of_the_line_does_not_intersect() -> None:
    line = ((0.0, 0.0), (10.0, 0.0))
    step = ((20.0, -4.0), (20.0, 4.0))

    assert not segments_intersect(step, line)


def test_point_in_polygon_includes_the_interior_and_the_boundary() -> None:
    assert point_in_polygon((5.0, 5.0), SQUARE)
    assert point_in_polygon((0.0, 0.0), SQUARE)
    assert point_in_polygon((10.0, 5.0), SQUARE)
    assert not point_in_polygon((15.0, 5.0), SQUARE)
    assert not point_in_polygon((-1.0, 5.0), SQUARE)


def test_a_concave_polygon_does_not_count_the_bay() -> None:
    c_shape = (
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 3.0),
        (3.0, 3.0),
        (3.0, 7.0),
        (10.0, 7.0),
        (10.0, 10.0),
        (0.0, 10.0),
    )

    assert point_in_polygon((1.0, 5.0), c_shape)
    assert not point_in_polygon((7.0, 5.0), c_shape)


def test_a_two_vertex_polygon_contains_nothing() -> None:
    assert not point_in_polygon((1.0, 1.0), ((0.0, 0.0), (2.0, 2.0)))


def test_distance_and_polyline_length() -> None:
    assert euclidean_distance((0.0, 0.0), (3.0, 4.0)) == pytest.approx(5.0)
    assert polyline_length(((0.0, 0.0), (3.0, 0.0), (3.0, 4.0))) == pytest.approx(7.0)
    assert polyline_length(((1.0, 1.0),)) == 0.0
