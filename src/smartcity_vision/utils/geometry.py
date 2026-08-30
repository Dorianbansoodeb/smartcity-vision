"""2-D geometry used by line crossing, zones, and speed.

OpenCV's ``pointPolygonTest`` is the source of truth for inclusion; line
crossing uses the standard orientation test so a crossing is defined without
depending on floating-point pixel rasterisation.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

Point = tuple[float, float]
Segment = tuple[Point, Point]
Polygon = tuple[Point, ...]

_EPSILON = 1e-9


def side_of_line(point: Point, start: Point, end: Point) -> int:
    """Return which side of the directed segment ``start→end`` ``point`` is on.

    Returns:
        ``+1`` if ``point`` is to the left of the directed line, ``-1`` if to
        the right, ``0`` if it lies on the line (within a small epsilon).
    """
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )
    if abs(cross) <= _EPSILON:
        return 0
    return 1 if cross > 0 else -1


def segments_intersect(first: Segment, second: Segment) -> bool:
    """Whether two closed line segments intersect, including endpoints.

    Uses the orientation method: the segments intersect if each straddles the
    other's supporting line, or if they are collinear and their projections
    overlap. A shared endpoint counts as an intersection so a trajectory that
    lands exactly on a counting line is not silently dropped.
    """
    a, b = first
    c, d = second
    o1 = side_of_line(c, a, b)
    o2 = side_of_line(d, a, b)
    o3 = side_of_line(a, c, d)
    o4 = side_of_line(b, c, d)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(c, a, b):
        return True
    if o2 == 0 and _on_segment(d, a, b):
        return True
    if o3 == 0 and _on_segment(a, c, d):
        return True
    return o4 == 0 and _on_segment(b, c, d)


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Whether ``point`` is inside ``polygon``, including the boundary.

    Delegates to OpenCV's ``pointPolygonTest`` so inclusion matches the same
    library that draws the overlay. A polygon with fewer than three vertices
    cannot enclose anything.

    Args:
        point: Query point in the same coordinate system as ``polygon``.
        polygon: Vertices in order, clockwise or counter-clockwise.

    Returns:
        ``True`` if the point is inside or on the edge.
    """
    if len(polygon) < 3:
        return False
    contour = np.asarray(polygon, dtype=np.float32)
    return float(cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)) >= 0.0


def euclidean_distance(first: Point, second: Point) -> float:
    """Euclidean distance between two points, in the same units as the inputs."""
    return float(np.hypot(second[0] - first[0], second[1] - first[1]))


def polyline_length(points: Sequence[Point]) -> float:
    """Sum of consecutive segment lengths along ``points``."""
    if len(points) < 2:
        return 0.0
    return float(
        sum(
            euclidean_distance(points[index], points[index + 1]) for index in range(len(points) - 1)
        )
    )


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    """Whether ``point`` lies on the closed segment ``start–end``.

    Callers must already have established collinearity via :func:`side_of_line`.
    """
    return (
        min(start[0], end[0]) - _EPSILON <= point[0] <= max(start[0], end[0]) + _EPSILON
        and min(start[1], end[1]) - _EPSILON <= point[1] <= max(start[1], end[1]) + _EPSILON
    )


__all__ = [
    "Point",
    "Polygon",
    "Segment",
    "euclidean_distance",
    "point_in_polygon",
    "polyline_length",
    "segments_intersect",
    "side_of_line",
]
