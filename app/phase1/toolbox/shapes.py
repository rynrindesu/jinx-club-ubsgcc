"""Deterministic classifier for the Tool-box PNG shape task."""

from __future__ import annotations

import base64
from collections import Counter
from io import BytesIO
from math import inf, hypot

from PIL import Image, UnidentifiedImageError


def classify_shape_image(image_base64: str) -> str:
    """Classify a filled PNG as a rectangle, triangle, or circle.

    The challenge images contain one high-contrast, filled shape on a uniform
    or transparent background.  The foreground's area relative to its minimum
    *oriented* bounding rectangle separates the three possible shapes without
    an ML model: a rectangle fills the box, a circle fills pi / 4 of it, and a
    triangle fills one half of it.
    """

    image = _decode_png(image_base64).convert("RGBA")
    background = _border_background(image)
    foreground = [
        _is_foreground(pixel, background)
        for pixel in image.getdata()
    ]

    width, height = image.size
    occupied = {
        (index % width, index // width)
        for index, present in enumerate(foreground)
        if present
    }
    if not occupied:
        raise ValueError("image does not contain a visible shape")

    # Ignore isolated foreground specks and retain the supplied image's one
    # visible shape. Eight-way connectivity keeps anti-aliased diagonal edges
    # in the same component.
    shape = _largest_component(occupied)
    hull = _convex_hull(shape)
    box_area = _minimum_oriented_bounding_box_area(hull)
    if box_area == 0:
        raise ValueError("image does not contain a two-dimensional shape")

    rectangularity = len(shape) / box_area
    if rectangularity >= 0.90:
        return "rectangle"
    if rectangularity <= 0.63:
        return "triangle"
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


def _largest_component(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return the largest eight-connected foreground component."""

    remaining = set(points)
    largest: list[tuple[int, int]] = []

    while remaining:
        start = remaining.pop()
        component = [start]
        pending = [start]
        while pending:
            x, y = pending.pop()
            for x_offset in (-1, 0, 1):
                for y_offset in (-1, 0, 1):
                    if x_offset == y_offset == 0:
                        continue
                    neighbour = (x + x_offset, y + y_offset)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        component.append(neighbour)
                        pending.append(neighbour)

        if len(component) > len(largest):
            largest = component

    return largest


def _minimum_oriented_bounding_box_area(hull: list[tuple[int, int]]) -> float:
    """Return the area of the smallest rectangle enclosing ``hull``.

    One side of a minimum-area bounding rectangle is collinear with a convex
    hull edge.  Testing every hull-edge orientation therefore finds the
    rectangle without requiring an additional imaging dependency.
    """

    if len(hull) < 3:
        return 0

    smallest_area = inf
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        edge_length = hypot(end[0] - start[0], end[1] - start[1])
        if edge_length == 0:
            continue

        along_x = (end[0] - start[0]) / edge_length
        along_y = (end[1] - start[1]) / edge_length
        across_x, across_y = -along_y, along_x
        along = [point[0] * along_x + point[1] * along_y for point in hull]
        across = [point[0] * across_x + point[1] * across_y for point in hull]
        area = (max(along) - min(along)) * (max(across) - min(across))
        smallest_area = min(smallest_area, area)

    return smallest_area
