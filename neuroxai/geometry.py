"""Measurements and view geometry in slice pixel coordinates.

A point is (x, y) in pixels: x is the column and y is the row of a slice. pixel_spacing is the
(horizontal, vertical) size of one pixel in mm (see Volume.pixel_spacing).
"""

import math

MAX_TICKS = 15


def distance_mm(p1, p2, pixel_spacing):
    """Length of the segment p1-p2 in mm."""
    sx, sy = pixel_spacing
    return math.hypot((p2[0] - p1[0]) * sx, (p2[1] - p1[1]) * sy)


def polygon_area(points):
    """Area of a simple polygon in square pixels (shoelace formula). The last point connects to the first."""
    if len(points) < 3:
        return 0.0
    twice_area = 0.0
    for (x1, y1), (x2, y2) in zip(points, [*points[1:], points[0]], strict=True):
        twice_area += x1 * y2 - x2 * y1
    return abs(twice_area) / 2.0


def area_mm2(points, pixel_spacing):
    """Area of a simple polygon in mm²."""
    return polygon_area(points) * pixel_spacing[0] * pixel_spacing[1]


def nice_step(value):
    """The smallest number of the form (1, 2, 2.5 or 5) x 10^n that is not less than value."""
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if step >= value * (1 - 1e-9):
            return step
    return 10 * magnitude


def ticks(low, high, mm_per_pixel, count):
    """Pixel positions in [low, high] of about count ticks at a nice interval in mm (at most MAX_TICKS)."""
    step_mm = nice_step((high - low) * mm_per_pixel / count)
    position_mm = math.ceil(low * mm_per_pixel / step_mm) * step_mm
    positions = []
    while position_mm / mm_per_pixel <= high + 1e-9 and len(positions) < MAX_TICKS:
        positions.append(position_mm / mm_per_pixel)
        position_mm += step_mm
    return positions


def format_mm(value):
    """Tick label for a position in mm: fewer decimals for larger values."""
    magnitude = abs(value)
    if magnitude == 0:
        return "0"
    for limit, decimals in ((0.001, 4), (0.01, 3), (0.1, 2), (10, 1)):
        if magnitude < limit:
            return f"{value:.{decimals}f}"
    return f"{value:.0f}"


def zoom_limits(size, zoom, center):
    """(low, high) limits of one view axis over an image axis of size pixels.

    zoom is the magnification (1 shows the full axis) and center is the center of the view as a
    fraction of size. Near an edge, the window moves so that it stays inside the image.
    """
    width = min(size, size / zoom)
    low = min(max(center * size - width / 2, 0.0), size - width)
    return low, low + width
