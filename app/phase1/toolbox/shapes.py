"""Deterministic classifier for the Tool-box PNG shape task."""

from __future__ import annotations

import base64
from collections import Counter
from io import BytesIO
from math import hypot

from PIL import Image, UnidentifiedImageError


def classify_shape_image(image_base64: str) -> str:
    """Classify a filled PNG as a rectangle, triangle, or circle.

    The challenge images contain one high-contrast shape on a uniform or
    transparent background.  Comparing the foreground area to its bounding
    box distinguishes the three possible filled shapes without an ML model.
    """

    image = _decode_png(image_base64).convert("RGBA")
    background = _border_background(image)
    foreground = [
        _is_foreground(pixel, background)
        for pixel in image.getdata()
    ]

    width, height = image.size
    occupied = [
        (index % width, index // width)
        for index, present in enumerate(foreground)
        if present
    ]
    if not occupied:
        raise ValueError("image does not contain a visible shape")

    corners = _simplify_hull(_convex_hull(occupied))
    edge_count = len(corners)
    if edge_count == 3:
        return "triangle"
    if edge_count == 4:
        return "rectangle"
    return "circle"


def _decode_png(image_base64: str) -> Image.Image:
    encoded = image_base64.strip()
    if encoded.startswith("data:"):
        try:
            encoded = encoded.split(",", 1)[1]
        except IndexError as error:
            raise ValueError("invalid data URI") from error

    try:
        raw_image = base64.b64decode(encoded, validate=True)
        image = Image.open(BytesIO(raw_image))
        image.load()
    except (ValueError, UnidentifiedImageError, OSError) as error:
        raise ValueError("image_base64 must be a valid base64-encoded PNG") from error

    if image.format != "PNG":
        raise ValueError("image_base64 must contain a PNG image")
    return image


def _border_background(image: Image.Image) -> tuple[int, int, int, int]:
    width, height = image.size
    pixels = image.load()
    border = [pixels[x, 0] for x in range(width)]
    border.extend(pixels[x, height - 1] for x in range(width))
    border.extend(pixels[0, y] for y in range(height))
    border.extend(pixels[width - 1, y] for y in range(height))
    return Counter(border).most_common(1)[0][0]


def _is_foreground(
    pixel: tuple[int, int, int, int],
    background: tuple[int, int, int, int],
) -> bool:
    if background[3] <= 16:
        return pixel[3] > 16
    return max(abs(component - reference) for component, reference in zip(pixel, background)) > 16


def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the outer boundary vertices in counter-clockwise order."""

    ordered = sorted(set(points))
    if len(ordered) <= 2:
        return ordered

    def cross(
        origin: tuple[int, int],
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> int:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[int, int]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[int, int]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _simplify_hull(hull: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Approximate a rasterized hull with its meaningful straight sides."""

    if len(hull) <= 4:
        return hull

    start_index = min(range(len(hull)), key=lambda index: hull[index])
    start = hull[start_index]
    end_index = max(
        range(len(hull)),
        key=lambda index: _squared_distance(start, hull[index]),
    )
    end = hull[end_index]

    first_arc = _cyclic_slice(hull, start_index, end_index)
    second_arc = _cyclic_slice(hull, end_index, start_index)
    width = max(point[0] for point in hull) - min(point[0] for point in hull)
    height = max(point[1] for point in hull) - min(point[1] for point in hull)
    # A four-percent tolerance absorbs the paired corners introduced by a
    # thick anti-aliased outline while retaining the visibly distinct sides
    # of a triangle or rectangle.
    tolerance = max(width, height) * 0.04

    # Each arc includes both endpoints.  Dropping the duplicated final point
    # from each produces one closed polygon with unique vertices.
    return _simplify_path(first_arc, tolerance)[:-1] + _simplify_path(second_arc, tolerance)[:-1]


def _cyclic_slice(
    points: list[tuple[int, int]],
    start_index: int,
    end_index: int,
) -> list[tuple[int, int]]:
    if start_index <= end_index:
        return points[start_index : end_index + 1]
    return points[start_index:] + points[: end_index + 1]


def _simplify_path(
    points: list[tuple[int, int]],
    tolerance: float,
) -> list[tuple[int, int]]:
    if len(points) <= 2:
        return points

    start, end = points[0], points[-1]
    index, distance = max(
        (
            (index, _distance_to_line(point, start, end))
            for index, point in enumerate(points[1:-1], start=1)
        ),
        key=lambda item: item[1],
    )
    if distance <= tolerance:
        return [start, end]

    return (
        _simplify_path(points[: index + 1], tolerance)[:-1]
        + _simplify_path(points[index:], tolerance)
    )


def _distance_to_line(
    point: tuple[int, int],
    start: tuple[int, int],
    end: tuple[int, int],
) -> float:
    line_length = hypot(end[0] - start[0], end[1] - start[1])
    if line_length == 0:
        return hypot(point[0] - start[0], point[1] - start[1])
    return abs(
        (end[0] - start[0]) * (start[1] - point[1])
        - (start[0] - point[0]) * (end[1] - start[1])
    ) / line_length


def _squared_distance(first: tuple[int, int], second: tuple[int, int]) -> int:
    return (second[0] - first[0]) ** 2 + (second[1] - first[1]) ** 2
