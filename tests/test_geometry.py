import math

import pytest

from neuroxai.geometry import (
    area_mm2,
    distance_mm,
    format_intensity,
    format_mm,
    nice_step,
    polygon_area,
    ticks,
    zoom_limits,
)


def test_distance_uses_the_spacing_of_each_axis():
    assert distance_mm((0, 0), (3, 4), (1.0, 1.0)) == pytest.approx(5.0)
    assert distance_mm((1, 1), (4, 5), (2.0, 0.5)) == pytest.approx(math.hypot(6.0, 2.0))


def test_polygon_area_closes_the_polygon():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert polygon_area(square) == pytest.approx(100.0)
    assert polygon_area([*square, (0, 0)]) == pytest.approx(100.0)
    assert polygon_area(square[::-1]) == pytest.approx(100.0)
    assert polygon_area([(0, 0), (4, 0), (0, 3)]) == pytest.approx(6.0)
    assert polygon_area([(0, 0), (1, 1)]) == 0.0
    assert area_mm2(square, (1.2, 1.5)) == pytest.approx(180.0)


@pytest.mark.parametrize(
    ("value", "step"), [(0.7, 1), (1, 1), (1.1, 2), (2.2, 2.5), (3, 5), (7, 10), (12, 20), (0.03, 0.05), (0, 1)]
)
def test_nice_step(value, step):
    assert nice_step(value) == pytest.approx(step)


def test_ticks_are_at_nice_mm_positions_inside_the_range():
    positions = ticks(3.0, 97.0, 1.2, 6)  # 3.6 mm to 116.4 mm, a step of 20 mm
    assert [round(p * 1.2, 6) for p in positions] == [20, 40, 60, 80, 100]
    assert len(ticks(0, 10_000, 0.001, 6)) <= 15


@pytest.mark.parametrize(
    ("value", "text"), [(0, "0"), (0.0005, "0.0005"), (0.005, "0.005"), (0.05, "0.05"), (5.25, "5.2"), (123.4, "123")]
)
def test_format_mm(value, text):
    assert format_mm(value) == text


def test_zoom_limits_stay_inside_the_image():
    assert zoom_limits(100, 1.0, 0.5) == (0.0, 100.0)
    assert zoom_limits(100, 0.5, 0.2) == (0.0, 100.0)
    assert zoom_limits(100, 4.0, 0.5) == pytest.approx((37.5, 62.5))
    assert zoom_limits(100, 4.0, 0.0) == pytest.approx((0.0, 25.0))
    assert zoom_limits(100, 4.0, 1.0) == pytest.approx((75.0, 100.0))


def test_format_intensity():
    assert [format_intensity(v) for v in (584.4, -1024.0, 0.5, 1.0, 12.345)] == ["584", "-1024", "0.5", "1", "12.3"]
