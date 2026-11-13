"""The scikit-image 0.25 fallback of spaCR's size and hole thresholds.

An older scikit-image has no ``max_size`` keyword; the fallback call must
keep spaCR's ``<`` boundary so an object or hole of exactly the threshold
survives.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")

import spacr.object as O
from spacr import measure as _measure


# ---------------------------------------------------------------------------
# scikit-image 0.25 fallback
# ---------------------------------------------------------------------------

def _old_remove_small_objects(ar, min_size=64, connectivity=1, *, out=None):
    """scikit-image 0.25's ``remove_small_objects``: no ``max_size`` keyword."""
    from skimage.measure import label as sk_label
    labelled = sk_label(np.asarray(ar), connectivity=connectivity)
    keep = np.zeros(np.asarray(ar).shape, dtype=bool)
    for value in np.unique(labelled):
        if value == 0:
            continue
        component = labelled == value
        if component.sum() >= min_size:
            keep |= component
    return keep


def _old_remove_small_holes(ar, area_threshold=64, connectivity=1, *, out=None):
    """scikit-image 0.25's ``remove_small_holes``: no ``max_size`` keyword."""
    filled = np.asarray(ar).copy()
    holes = ~np.asarray(ar)
    from skimage.measure import label as sk_label
    labelled = sk_label(holes, connectivity=connectivity)
    for value in np.unique(labelled):
        if value == 0:
            continue
        component = labelled == value
        if component.sum() < area_threshold:
            filled |= component
    return filled


def test_the_old_skimage_call_keeps_the_same_size_boundary(monkeypatch):
    """On 0.25 the fallback must pass min_size, not the decremented threshold.

    0.26 removes objects strictly SMALLER than ``max_size`` + 1, so spaCR
    passes ``min_size - 1``. Handing that same number to the old keyword would
    delete every object of exactly ``min_size`` px -- objects the user asked to
    keep.
    """
    seen = {}

    def _recording(ar, min_size=64, connectivity=1, *, out=None):
        seen["min_size"] = min_size
        return _old_remove_small_objects(ar, min_size, connectivity, out=out)

    monkeypatch.setattr(_measure.morphology, "remove_small_objects", _recording)

    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True       # area 4
    mask[5, 5:7] = True         # area 2
    result = O._remove_objects_smaller_than(mask, 4)

    assert seen["min_size"] == 4
    assert result[1:3, 1:3].all(), "an object of exactly min_size is kept"
    assert not result[5, 5:7].any()


def test_the_old_skimage_hole_call_keeps_the_same_area_boundary(monkeypatch):
    """The hole filler's fallback keeps a hole of exactly area_threshold."""
    seen = {}

    def _recording(ar, area_threshold=64, connectivity=1, *, out=None):
        seen["area_threshold"] = area_threshold
        return _old_remove_small_holes(ar, area_threshold, connectivity, out=out)

    monkeypatch.setattr(_measure.morphology, "remove_small_holes", _recording)

    mask = np.ones((10, 10), dtype=bool)
    mask[2:4, 2:4] = False      # area 4
    mask[7, 7:9] = False        # area 2
    result = O._fill_holes_smaller_than(mask, 4)

    assert seen["area_threshold"] == 4
    assert not result[2:4, 2:4].any(), "a hole of exactly the threshold stays"
    assert result[7, 7:9].all()
