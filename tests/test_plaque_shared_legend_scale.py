"""A legend-described scale bar applies to its same-size plaque grid."""
from dataclasses import replace

import numpy as np
import pytest

from spacr import plaque_papers as pp


def _grid():
    image = np.full((240, 240, 3), 255, np.uint8)
    regions = [pp.Region(x, y, x + 100, y + 100)
               for y in (10, 115) for x in (10, 115)]
    for region in regions:
        image[region.y0:region.y1, region.x0:region.x1] = 128
    return image, regions


def _draw_bar(image, region, length=20, colour=255):
    image[region.y1 - 12:region.y1 - 9, region.x1 - 10 - length:region.x1 - 10] = colour


@pytest.mark.parametrize("bar_index", [0, 3])
@pytest.mark.parametrize("colour", [0, 255])
@pytest.mark.parametrize("reverse", [False, True])
def test_legend_ruler_reaches_every_equal_size_crop_independent_of_order(
        bar_index, colour, reverse):
    image, regions = _grid()
    owner = regions[bar_index]
    _draw_bar(image, owner, colour=colour)
    if reverse:
        regions.reverse()
    scales = pp._scales_for_regions(image, regions, [], caption="Plaques. Scale bar = 2 mm.")
    assert all(scale.px_per_mm == pytest.approx(10) for scale in scales)
    for region, scale in zip(regions, scales):
        assert scale.unit == "mm2"
        assert scale.source == ("scale bar, length from legend" if region == owner
                                else "scale bar, same panel")


def test_a_crops_own_unlabelled_bar_wins_over_a_neighbours_labelled_bar():
    image, regions = _grid()
    _draw_bar(image, regions[0], length=20, colour=0)
    _draw_bar(image, regions[1], length=40, colour=0)
    word = pp.Word("2 mm", 68, 75, 100, 86)
    scales = pp._scales_for_regions(image, regions[:2], [word], caption="Scale bar = 2 mm.")
    assert scales[0].px_per_mm == pytest.approx(10)
    assert scales[0].source == "scale bar"
    assert scales[1].px_per_mm == pytest.approx(20)
    assert scales[1].source == "scale bar, length from legend"


@pytest.mark.parametrize("kind", ["different_size", "separate_grid"])
def test_a_legend_ruler_is_not_shared_with_unrelated_crops(kind):
    image, regions = _grid()
    _draw_bar(image, regions[0])
    other = (replace(regions[1], x1=regions[1].x0 + 60) if kind == "different_size"
             else pp.Region(500, 500, 600, 600))
    first, second = pp._scales_for_regions(image, [regions[0], other], [],
                                         caption="Scale bar = 2 mm.")
    assert first.px_per_mm == pytest.approx(10)
    assert second.px_per_mm is None and second.unit == "px"


def test_an_unlabelled_bar_without_a_physical_length_does_not_invent_units():
    image, regions = _grid()
    _draw_bar(image, regions[3])
    scales = pp._scales_for_regions(image, regions, [], caption="Plaques, 10x objective.")
    assert all(scale.px_per_mm is None and scale.unit == "px" for scale in scales)


@pytest.mark.parametrize("second_length", [20, 40])
def test_neighbours_must_agree_before_their_scale_is_shared(second_length):
    image, regions = _grid()
    _draw_bar(image, regions[0], length=20)
    _draw_bar(image, regions[1], length=second_length)
    scales = pp._scales_for_regions(image, regions, [], caption="Scale bar = 2 mm.")
    assert scales[0].px_per_mm == pytest.approx(10)
    assert scales[1].px_per_mm == pytest.approx(second_length / 2)
    if second_length == 20:
        assert all(scale.px_per_mm == pytest.approx(10) for scale in scales)
    else:
        assert all(scale.px_per_mm is None and scale.unit == "px" for scale in scales[2:])
        assert all("conflicting" in scale.detail for scale in scales[2:])
