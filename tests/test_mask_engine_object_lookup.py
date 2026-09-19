"""The objects of a mask, measured the way the Make Masks filter measures them.

Item 419, point 1 (the maintainer, 2026-09-16): "In make masks, hovering
should show intensity and object area, etc." The readout is only worth having
if its numbers are the ones the size/intensity filter judges by, so these
tests hold :class:`spacr.qt.mask_engine.ObjectLookup` to the filter's own
arithmetic, and hold the faster :func:`spacr.qt.mask_engine.canonical_labels`
that made a per-mouse-move readout affordable to the algorithm it replaced.
No display is needed: mask_engine has no Qt dependency.
"""
from __future__ import annotations

import numpy as np

from spacr.qt import mask_engine as engine


def _field():
    """A 64 x 64 coded field and a mask whose id 9 is in two pieces."""
    yy, xx = np.mgrid[0:64, 0:64]
    image = (yy * 64 + xx).astype(np.uint16)
    mask = np.zeros((64, 64), np.uint16)
    mask[4:12, 4:14] = 3
    mask[20:36, 30:42] = 5
    mask[44:58, 50:60] = 9
    mask[50:53, 6:9] = 9
    return image, mask


def test_the_lookup_measures_exactly_what_the_filter_measures():
    """Every object, compared with the regionprops call the filter makes."""
    from skimage.measure import regionprops

    image, mask = _field()
    lookup = engine.ObjectLookup(mask, image)
    labels = engine.canonical_labels(mask)
    regions = regionprops(labels.astype(np.int32),
                          intensity_image=np.asarray(image, np.float32))
    assert len(regions) == 4
    for region in regions:
        assert lookup.measure(region.label) == (
            int(region.area), float(region.intensity_mean))


def test_a_bound_at_the_mean_keeps_the_object_and_one_past_it_does_not():
    image, mask = _field()
    readout = engine.ObjectLookup(mask, image).at(8, 8)
    assert readout.label == 3
    kept, dropped = engine.filter_objects(
        mask, image, min_intensity=readout.mean_intensity)
    assert 3 not in dropped and (kept == 3).any()
    _out, dropped = engine.filter_objects(
        mask, image, min_intensity=np.nextafter(readout.mean_intensity, np.inf))
    assert 3 in dropped
    _out, dropped = engine.filter_objects(mask, image, min_area=readout.area)
    assert 3 not in dropped
    _out, dropped = engine.filter_objects(mask, image, min_area=readout.area + 1)
    assert 3 in dropped


def test_a_pixel_reads_out_its_raw_value_and_its_object():
    image, mask = _field()
    lookup = engine.ObjectLookup(mask, image)
    readout = lookup.at(35, 25)
    assert readout == engine.PixelReadout(
        35, 25, float(25 * 64 + 35), 5, 16 * 12,
        float(np.mean(image[20:36, 30:42].astype(np.float32))))
    assert lookup.at(20, 40) == engine.PixelReadout(20, 40, float(40 * 64 + 20))
    assert lookup.at(64, 0) is None and lookup.at(-1, 3) is None


def test_the_small_piece_of_a_split_id_is_its_own_object():
    image, mask = _field()
    lookup = engine.ObjectLookup(mask, image)
    piece = lookup.at(7, 51)
    assert piece.label not in (0, 3, 5, 9)
    assert piece.area == 9
    assert lookup.at(55, 50).label == 9


def test_an_empty_mask_reads_out_the_pixel_only():
    image, _mask = _field()
    lookup = engine.ObjectLookup(np.zeros((64, 64), np.uint16), image)
    assert lookup.at(3, 3) == engine.PixelReadout(3, 3, float(3 * 64 + 3))
    assert lookup.measure(1) is None


def test_canonical_labels_is_unchanged_by_the_bounding_box_rewrite():
    """The fast path, against the whole-field algorithm it replaced."""
    from scipy import ndimage

    eight = np.ones((3, 3), np.uint8)

    def whole_field(mask):
        m = np.asarray(mask)
        values = np.unique(m[m > 0])
        if values.size <= 1:
            return ndimage.label(m > 0, structure=eight)[0].astype(np.uint16)
        out = m.astype(np.int64, copy=True)
        used = {int(v) for v in values}
        candidate = 1
        for value in values:
            pieces, count = ndimage.label(m == value, structure=eight)
            if count <= 1:
                continue
            keep = int(np.argmax(np.bincount(pieces.ravel())[1:])) + 1
            for piece in range(1, count + 1):
                if piece == keep:
                    continue
                while candidate in used:
                    candidate += 1
                out[pieces == piece] = candidate
                used.add(candidate)
        return out.astype(np.uint16)

    rng = np.random.default_rng(419)
    checked = 0
    for trial in range(60):
        shape = tuple(int(v) for v in rng.integers(8, 70, 2))
        dtype = (np.uint8, np.uint16, np.int32)[trial % 3]
        n_values = int(rng.integers(1, 9))
        mask = np.zeros(shape, dtype)
        for _blob in range(int(rng.integers(1, 25))):
            y, x = (int(rng.integers(0, s)) for s in shape)
            h, w = (int(rng.integers(1, 7)) for _ in range(2))
            mask[y:y + h, x:x + w] = int(rng.integers(1, n_values + 1)) * (
                31 if dtype is np.uint8 else 997)
        assert np.array_equal(engine.canonical_labels(mask),
                              whole_field(mask)), (trial, dtype)
        checked += 1
    assert checked == 60
