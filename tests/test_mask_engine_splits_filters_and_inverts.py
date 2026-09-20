"""The three engine operations item 419 part C is built on, with no display.

Point 7 needs the filter to say WHY it removed each object, point 8 needs an
object cut at its waist by a single click, and point 9 needs an inversion
that a detector can read. All three are in :mod:`spacr.qt.mask_engine` and
none of them needs Qt, so they are measured here on arrays where the right
answer is known by construction.

Two of these contracts are the ones a later change is most likely to break
quietly:

* ``filter_report`` must agree with ``filter_objects`` about every id, since
  the second is now the first with the reasons dropped. A divergence would
  put one set of objects in the mask and a different set in the ledger the
  user reads.
* ``split_object_at`` must not lose a pixel. A split that quietly erased the
  smaller half would be a delete wearing a split's name, and on a field of
  four hundred objects nobody would notice which.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt import mask_engine as engine


def peanut(shape=(40, 60), label=5, centres=((20, 20), (36, 20)), radius=10):
    """Two overlapping discs under ONE id: a pair that merged."""
    out = np.zeros(shape, dtype=np.uint16)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for cx, cy in centres:
        out[((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2] = label
    return out


def disc(shape=(40, 40), label=3, centre=(20, 20), radius=9):
    """One round object, which has one centre and no waist."""
    out = np.zeros(shape, dtype=np.uint16)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    out[((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) < radius ** 2] = label
    return out


# ---------------------------------------------------------------------------
# Point 7: the filter says which bound removed each object
# ---------------------------------------------------------------------------

@pytest.fixture
def field():
    """Three objects with known areas and known means."""
    mask = np.zeros((40, 40), dtype=np.uint16)
    image = np.zeros((40, 40), dtype=np.uint16)
    mask[2:5, 2:5] = 3
    image[2:5, 2:5] = 200
    mask[10:20, 10:20] = 4
    image[10:20, 10:20] = 800
    mask[24:30, 24:30] = 9
    image[24:30, 24:30] = 4000
    return mask, image


def test_a_removal_names_the_object_its_area_its_mean_and_the_bound(field):
    mask, image = field
    _out, removals = engine.filter_report(mask, image, min_area=20)
    assert len(removals) == 1
    removal = removals[0]
    assert (removal.label, removal.area) == (3, 9)
    assert removal.mean_intensity == pytest.approx(200.0)
    assert removal.bounds == ("min_area",)


def test_an_object_outside_two_bounds_names_both(field):
    mask, image = field
    _out, removals = engine.filter_report(
        mask, image, min_area=20, min_intensity=300.0)
    assert [(r.label, r.bounds) for r in removals] == [
        (3, ("min_area", "min_intensity"))]


def test_each_bound_can_be_the_one_that_removes(field):
    mask, image = field
    for bound, kwargs, expected in (
            ("min_area", {"min_area": 20}, [3]),
            ("max_area", {"max_area": 50}, [4]),
            ("min_intensity", {"min_intensity": 500.0}, [3]),
            ("max_intensity", {"max_intensity": 1000.0}, [9]),
    ):
        _out, removals = engine.filter_report(mask, image, **kwargs)
        assert [r.label for r in removals] == expected, bound
        assert all(bound in r.bounds for r in removals), bound


def test_the_report_and_the_plain_filter_agree_on_every_id(field):
    mask, image = field
    bounds = {"min_area": 20, "max_intensity": 3000.0}
    plain, dropped = engine.filter_objects(mask, image, **bounds)
    reported, removals = engine.filter_report(mask, image, **bounds)
    assert dropped == [r.label for r in removals]
    np.testing.assert_array_equal(plain, reported)


def test_nothing_to_do_returns_the_array_itself_and_no_rows(field):
    mask, image = field
    out, removals = engine.filter_report(mask, image)
    assert out is mask and removals == []
    out, removals = engine.filter_report(mask, image, min_area=2)
    assert out is mask and removals == []


def test_a_removals_area_and_mean_are_the_readouts_own(field):
    """The row a user reads names the numbers the corner readout showed."""
    mask, image = field
    lookup = engine.ObjectLookup(mask, image)
    _out, removals = engine.filter_report(mask, image, min_intensity=5000.0)
    for removal in removals:
        assert lookup.measure(removal.label) == (
            removal.area, pytest.approx(removal.mean_intensity))


# ---------------------------------------------------------------------------
# Point 8: a click cuts the object at its waist
# ---------------------------------------------------------------------------

def test_a_merged_pair_comes_apart_at_its_waist():
    mask = peanut()
    out, new_ids = engine.split_object_at(mask, 20, 20)
    assert len(new_ids) == 1
    assert int(out[20, 20]) != int(out[20, 36])
    assert {int(out[20, 20]), int(out[20, 36])} == {5, new_ids[0]}


def test_the_split_loses_no_pixel():
    mask = peanut()
    out, _new = engine.split_object_at(mask, 20, 20)
    np.testing.assert_array_equal(out > 0, mask > 0)


def test_the_bigger_half_keeps_the_id():
    """The rule :func:`canonical_labels` already applies to one id, two blobs."""
    mask = np.zeros((40, 60), dtype=np.uint16)
    yy, xx = np.mgrid[0:40, 0:60]
    mask[((xx - 18) ** 2 + (yy - 20) ** 2) < 121] = 5
    mask[((xx - 38) ** 2 + (yy - 20) ** 2) < 36] = 5
    out, new_ids = engine.split_object_at(mask, 18, 20)
    assert new_ids, "the two lobes should have come apart"
    assert int(np.count_nonzero(out == 5)) > int(
        np.count_nonzero(out == new_ids[0]))
    assert int(out[20, 18]) == 5


def test_a_new_id_is_above_the_masks_top_label():
    mask = peanut()
    mask[0:3, 0:3] = 40
    _out, new_ids = engine.split_object_at(mask, 20, 20)
    assert new_ids == [41]


def test_an_object_with_one_centre_is_left_alone():
    mask = disc()
    out, new_ids = engine.split_object_at(mask, 20, 20)
    assert new_ids == []
    np.testing.assert_array_equal(out, mask)


def test_background_and_the_outside_change_nothing():
    mask = peanut()
    for x, y in ((1, 1), (-1, 5), (500, 5), (5, 500)):
        out, new_ids = engine.split_object_at(mask, x, y)
        assert new_ids == []
        np.testing.assert_array_equal(out, mask)


def test_min_area_decides_what_is_too_small_to_be_two():
    """A pair closer together than the smallest object is one object."""
    mask = peanut(centres=((20, 20), (30, 20)), radius=7)
    assert engine.split_object_at(mask, 20, 20, min_area=0)[1]
    assert engine.split_object_at(mask, 20, 20, min_area=4_000)[1] == []


def test_min_area_never_erases_the_smaller_half():
    """It sets the seed spacing; it does not drop what it spaced."""
    mask = peanut()
    out, new_ids = engine.split_object_at(mask, 20, 20, min_area=200)
    if new_ids:
        np.testing.assert_array_equal(out > 0, mask > 0)


def test_the_dtype_widens_only_when_the_new_id_needs_it():
    mask = peanut().astype(np.uint8)
    mask[mask > 0] = 250
    out, new_ids = engine.split_object_at(mask, 20, 20)
    assert new_ids == [251] and out.dtype == np.uint8


# ---------------------------------------------------------------------------
# Point 9: the inversion a detector reads
# ---------------------------------------------------------------------------

def test_the_darkest_pixel_becomes_the_brightest_and_back():
    values = np.array([[10, 40, 90]], dtype=np.uint16)
    inverted = engine.invert_intensity(values)
    np.testing.assert_array_equal(inverted, [[90, 60, 10]])
    np.testing.assert_array_equal(engine.invert_intensity(inverted), values)


def test_the_span_is_the_one_it_started_with():
    values = (np.arange(64, dtype=np.uint16) * 37).reshape(8, 8)
    inverted = engine.invert_intensity(values)
    assert (int(inverted.min()), int(inverted.max())) == (
        int(values.min()), int(values.max()))


def test_it_is_not_the_dtypes_maximum():
    """A 12-bit field in a uint16 array stays where the user could see it."""
    values = np.array([[0, 2048, 4095]], dtype=np.uint16)
    assert int(engine.invert_intensity(values).max()) == 4095


def test_given_bounds_it_reflects_about_those_instead():
    """What a crop is handed, so the box inverts as the whole field does."""
    whole = np.array([[0, 100, 200, 300]], dtype=np.uint16)
    crop = whole[:, 1:3]
    bounds = (float(whole.min()), float(whole.max()))
    np.testing.assert_array_equal(
        engine.invert_intensity(crop, bounds=bounds), [[200, 100]])
    np.testing.assert_array_equal(
        engine.invert_intensity(crop), [[200, 100]])
    other = np.array([[0, 50]], dtype=np.uint16)
    np.testing.assert_array_equal(
        engine.invert_intensity(other, bounds=bounds), [[300, 250]])


def test_the_original_is_never_touched():
    values = np.array([[10, 40, 90]], dtype=np.uint16)
    before = values.copy()
    engine.invert_intensity(values)
    np.testing.assert_array_equal(values, before)


def test_the_dtype_survives():
    for dtype in (np.uint8, np.uint16, np.int32, np.float32):
        values = np.array([[1, 5, 9]], dtype=dtype)
        assert engine.invert_intensity(values).dtype == dtype


def test_a_nonfinite_pixel_is_neither_dark_nor_bright():
    values = np.array([[0.0, np.nan, 4.0]], dtype=np.float32)
    inverted = engine.invert_intensity(values)
    assert np.isnan(inverted[0, 1])
    np.testing.assert_array_equal(inverted[0, [0, 2]], [4.0, 0.0])


def test_an_empty_field_comes_back_empty():
    values = np.zeros((0, 0), dtype=np.uint16)
    assert engine.invert_intensity(values).shape == (0, 0)


def test_a_flat_field_is_unchanged_rather_than_undefined():
    values = np.full((4, 4), 7, dtype=np.uint16)
    np.testing.assert_array_equal(engine.invert_intensity(values), values)


def test_otsu_finds_a_dark_object_once_the_field_is_inverted():
    """Point 9c's reason for existing, on the engine's own detector."""
    field = np.full((64, 64), 50_000, dtype=np.uint16)
    yy, xx = np.mgrid[0:64, 0:64]
    field[((xx - 20) ** 2 + (yy - 20) ** 2) < 64] = 2_000
    field[((xx - 44) ** 2 + (yy - 44) ** 2) < 64] = 2_000

    plain = engine._otsu_instances(field, bright=True, min_area=20)
    inverted = engine._otsu_instances(
        engine.invert_intensity(field), bright=True, min_area=20)
    assert int(inverted[20, 20]) and int(inverted[44, 44])
    assert int(inverted[20, 20]) != int(inverted[44, 44])
    assert int(plain.max()) < int(inverted.max())
