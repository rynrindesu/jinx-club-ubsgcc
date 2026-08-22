"""Deterministic classifier for the Tool-box PNG shape task."""

from __future__ import annotations

import base64
from collections import Counter
from io import BytesIO

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

    min_x = min(x for x, _ in occupied)
    max_x = max(x for x, _ in occupied)
    min_y = min(y for _, y in occupied)
    max_y = max(y for _, y in occupied)
    bounding_area = (max_x - min_x + 1) * (max_y - min_y + 1)
    fill_ratio = len(occupied) / bounding_area

    # A filled rectangle occupies nearly its whole bounding box, a triangle
    # roughly half, and a circle about pi / 4.  The gaps leave room for
    # anti-aliased PNG edges.
    if fill_ratio >= 0.90:
        return "rectangle"
    if fill_ratio <= 0.64:
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
