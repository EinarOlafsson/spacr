"""The well is the ruler: detection geometry, and pixels to millimetres.

A plaque area in pixels is a property of the microscope. The same plaque at two
magnifications gives two areas, and a study that pools them is comparing
optics. The well is a manufactured object of known physical size that is
already in the image, so its measured diameter converts every area into mm^2.

That makes the arithmetic here load-bearing for the science rather than
cosmetic, which is why the conversion is driven with a KNOWN answer rather
than checked for self-consistency: a scale that is wrong by a constant factor
is self-consistent and silently rescales an entire study.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from spacr import plaque
from spacr.plaque import (WELL_DIAMETERS_MM, PlaqueScale, Well, crop_well,
                          scale_from_well)


def test_a_missing_well_detector_names_the_optional_dependency(monkeypatch):
    """No detector install must produce an actionable error, not a bare import."""
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(ImportError, match=r'pip install "spacr\[plaque\]"') as caught:
        plaque._load_detector("wells.pt")

    assert isinstance(caught.value.__cause__, ImportError)


def test_an_installed_well_detector_receives_the_requested_weights(monkeypatch):
    """The positive import path still constructs the detector exactly once."""
    calls = []

    class FakeYOLO:
        def __init__(self, weights):
            calls.append(weights)

    fake = types.ModuleType("ultralytics")
    fake.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    detector = plaque._load_detector("wells.pt")

    assert isinstance(detector, FakeYOLO)
    assert calls == ["wells.pt"]


def test_a_square_box_gives_its_side_as_the_diameter():
    well = Well(x0=100, y0=200, x1=400, y1=500)
    assert well.width == 300 and well.height == 300
    assert well.diameter_px == 300
    assert well.axis_ratio == 1.0


def test_a_lopsided_box_is_reported_as_lopsided():
    """The honesty check on the ruler.

    A well is round, so a box far from square means the detector clipped it at
    an image edge, merged two wells, or found something else. Any of those
    makes the diameter wrong, and the diameter rescales every area in that
    well -- so a wrong one is worse than a missing one.
    """
    well = Well(x0=0, y0=0, x1=400, y1=200)
    assert well.diameter_px == 300, "the mean degrades gently"
    assert well.axis_ratio == 0.5, "and the disagreement is still visible"


def test_a_zero_sized_box_has_no_axis_ratio_or_physical_scale():
    """Degenerate detector output cannot become a plausible ruler."""
    well = Well(x0=4, y0=4, x1=4, y1=4)

    assert well.axis_ratio == 0.0
    assert scale_from_well(well, well_diameter_mm=10.0) is None


def test_well_detection_filters_bad_boxes_and_orders_the_rest(
    monkeypatch, caplog
):
    """Detector output is parsed, filtered, and returned in reading order."""
    class Box:
        def __init__(self, xyxy, confidence):
            self.xyxy = np.asarray([xyxy], dtype=float)
            self.conf = confidence

    boxes = [
        Box((30, 20, 40, 30), np.asarray([0.8])),
        Box((4, 4, 4, 12), np.asarray([0.7])),
        Box((0, 0, 20, 5), np.asarray([0.6])),
        Box((10, 5, 20, 15), None),
    ]
    results = [types.SimpleNamespace(boxes=None), types.SimpleNamespace(boxes=boxes)]
    calls = []

    class Detector:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return results

    monkeypatch.setattr(plaque, "_load_detector", lambda weights: Detector())

    with caplog.at_level("WARNING"):
        wells = plaque.detect_wells(
            np.zeros((40, 40), dtype=np.uint8),
            "wells.pt",
            confidence=0.4,
            imgsz=320,
            min_axis_ratio=0.7,
        )

    assert [(well.x0, well.y0, well.confidence) for well in wells] == [
        (10, 5, 1.0),
        (30, 20, 0.8),
    ]
    assert len(calls) == 1
    assert np.array_equal(
        calls[0].pop("source"), np.zeros((40, 40), dtype=np.uint8)
    )
    assert calls[0] == {"conf": 0.4, "imgsz": 320, "verbose": False}
    assert "axis ratio 0.25 is below 0.70" in caplog.text


def test_the_conversion_is_right_against_a_worked_example():
    """A 6-well well is 34.8 mm. Measured at 348 px, that is 10 px/mm.

    At 10 px/mm, one square millimetre is 100 square pixels. So a 5000-pixel
    plaque is 50 mm^2, and that number is checked against arithmetic done
    here rather than against the implementation's own output.
    """
    well = Well(x0=0, y0=0, x1=348, y1=348)
    scale = scale_from_well(well, plate_format="6-well")
    assert scale is not None
    assert scale.px_per_mm == pytest.approx(10.0)
    assert scale.area_mm2(5000) == pytest.approx(50.0)


def test_the_same_plaque_at_two_magnifications_measures_the_same():
    """The entire point, stated as a test.

    One plaque, two microscopes. The second images at twice the magnification,
    so both the well and the plaque are twice the size in pixels and the
    plaque covers four times the pixel area. In millimetres they must agree --
    if they do not, pooling plates from two scopes is meaningless.
    """
    low = Well(x0=0, y0=0, x1=348, y1=348)
    high = Well(x0=0, y0=0, x1=696, y1=696)

    low_scale = scale_from_well(low, plate_format="6-well")
    high_scale = scale_from_well(high, plate_format="6-well")

    plaque_low_px = 5000
    plaque_high_px = 5000 * 4          # twice the linear size

    assert low_scale.area_mm2(plaque_low_px) == pytest.approx(
        high_scale.area_mm2(plaque_high_px))
    assert low_scale.px_per_mm * 2 == pytest.approx(high_scale.px_per_mm)


def test_no_physical_diameter_means_no_scale_rather_than_a_guess():
    """``None``, not a default plate format.

    Assuming a format would fill the mm^2 columns with confident numbers that
    are wrong by whatever the real plate was -- and wrong by a constant factor,
    so nothing downstream would look odd.
    """
    well = Well(x0=0, y0=0, x1=348, y1=348)
    assert scale_from_well(well) is None


def test_an_unknown_plate_format_is_refused_rather_than_guessed():
    well = Well(x0=0, y0=0, x1=348, y1=348)
    with pytest.raises(KeyError, match="unknown plate format"):
        scale_from_well(well, plate_format="5-well")


def test_an_explicit_diameter_overrides_the_format():
    """A custom plate, or a dish measured by hand, must win."""
    well = Well(x0=0, y0=0, x1=100, y1=100)
    scale = scale_from_well(well, plate_format="6-well", well_diameter_mm=50.0)
    assert scale.well_diameter_mm == 50.0
    assert scale.source == "explicit"
    assert scale.px_per_mm == pytest.approx(2.0)


def test_every_declared_plate_format_produces_a_usable_scale():
    """No format may be present but unusable."""
    well = Well(x0=0, y0=0, x1=500, y1=500)
    for name in WELL_DIAMETERS_MM:
        scale = scale_from_well(well, plate_format=name)
        assert isinstance(scale, PlaqueScale)
        assert scale.px_per_mm > 0
        assert scale.source == name


def test_padding_past_the_TOP_LEFT_edge_clips_instead_of_wrapping():
    """The lower bound is the one that matters, and it is not symmetric.

    Numpy already truncates a slice that runs past the END of an array, so
    clamping the upper bound is defensive rather than load-bearing -- removing
    it does not change any result, which a mutation confirmed.

    The START is different. ``image[-15:]`` is not empty and does not raise: it
    is the LAST fifteen rows. So a well near the top-left, padded, would be cut
    from the OPPOSITE CORNER of the image -- a real crop, of real tissue, of
    the wrong well, with nothing anywhere to indicate it. Its plaques would be
    counted against a condition they did not come from.
    """
    image = np.zeros((100, 100), dtype=np.uint8)
    image[0:20, 0:20] = 1          # the well we mean
    image[80:100, 80:100] = 9      # what a negative index would take instead

    well = Well(x0=5, y0=5, x1=20, y1=20)
    crop = crop_well(image, well, pad=20)

    assert crop.shape == (40, 40), "clipped at 0, not wrapped"
    assert 9 not in np.unique(crop), (
        "the crop came from the far corner: a negative start index wrapped")
    assert 1 in np.unique(crop)


def test_cropping_past_the_bottom_right_edge_is_simply_truncated():
    """Documented rather than asserted as a guarantee of ours: this is numpy's
    behaviour, and the test exists so a reader knows the asymmetry is known."""
    image = np.zeros((100, 100), dtype=np.uint8)
    well = Well(x0=90, y0=90, x1=100, y1=100)
    assert crop_well(image, well, pad=20).shape == (30, 30)


def test_a_crop_well_inside_the_image_gets_its_padding():
    image = np.zeros((100, 100), dtype=np.uint8)
    well = Well(x0=40, y0=40, x1=60, y1=60)
    assert crop_well(image, well, pad=5).shape == (30, 30)
    assert crop_well(image, well).shape == (20, 20)
