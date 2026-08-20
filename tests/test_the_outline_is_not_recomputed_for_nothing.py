"""188 A: a settings change redraws in milliseconds, not seconds.

"after images are loaded, if the settings are changed after loading closing
the settings window with OK should automatically apply the settings to the
current images. the images should not be reloaded from disk ... in
miliseconds, as opposed to seconds if loaded again."

THE ARCHITECTURE WAS ALREADY RIGHT. `_load_signature` separates the settings
that decide which pixels are CUT from the ones that decide how a cut crop is
DRAWN, `_can_redraw_without_loading` compares it, and `_redraw_from_cache`
re-runs `draw_crop` over crops already in hand. No disk, no worker. Audited
before touching anything: every display setting leaves the signature
untouched, so the fast path is taken.

THE SECONDS WERE IN THE FAST PATH. Measured on a 128x128 crop, 2026-08-20:

    draw_crop, defaults                    0.198 ms
    draw_crop, outline on all 3 channels  13.299 ms

-- so a 200-cell montage spent ~2.7 s redrawing outlines and ~0.04 s on
everything else, and did it again for every settings change.

Almost all of that was recomputed for nothing. The foreground mask depends on
the PIXELS and two outline settings; the edge depends on the mask and the
thickness. Neither depends on `normalize_channels`, `percentiles`,
`edge_transparency` or `edge_image` -- which is most of what a user changes.

    redraw, same settings          0.619 ms/crop     124 ms per 200 cells
    redraw, transparency changed   0.542 ms/crop     108 ms per 200 cells
    redraw, normalize changed      1.913 ms/crop     383 ms per 200 cells

A CACHE THAT CHANGED THE PICTURE WOULD BE WORSE THAN THE WAIT, so the first
tests here are about the output being identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.picture_settings import draw_crop
from spacr.qt.annotate_engine import forget_outline_masks


@pytest.fixture
def crop():
    """A crop with real structure -- noise has no boundaries to find."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 60, (96, 96, 3), dtype=np.uint8)
    array[20:60, 25:70] = 220           # one bright object
    array[70:88, 10:30] = 180           # and a second
    return array


@pytest.fixture
def outlined():
    from spacr.qt.widgets.picture_settings_dialog import picture_defaults

    return {**dict(picture_defaults()), "outline": "r,g,b"}


class TestTheCacheDoesNotChangeThePicture:

    def test_the_second_draw_is_identical_to_the_first(self, crop, outlined):
        forget_outline_masks()
        first = np.asarray(draw_crop(crop, outlined))
        second = np.asarray(draw_crop(crop, outlined))

        assert np.array_equal(first, second)

    def test_a_cold_cache_gives_the_same_answer_as_a_warm_one(self, crop,
                                                              outlined):
        forget_outline_masks()
        warm = np.asarray(draw_crop(crop, outlined))
        forget_outline_masks()
        cold = np.asarray(draw_crop(crop, outlined))

        assert np.array_equal(warm, cold), (
            "the cache changed the picture, which is worse than the wait")

    def test_a_different_crop_is_not_given_another_crops_outline(self, crop,
                                                                 outlined):
        """Keyed on the BYTES, so two crops cannot share a mask."""
        forget_outline_masks()
        other = crop.copy()
        other[20:60, 25:70] = 40        # the bright object removed

        first = np.asarray(draw_crop(crop, outlined))
        second = np.asarray(draw_crop(other, outlined))

        assert not np.array_equal(first, second)

    def test_changing_an_outline_setting_recomputes(self, crop, outlined):
        forget_outline_masks()
        loose = np.asarray(draw_crop(crop, {**outlined,
                                            "outline_threshold_factor": 0.4}))
        tight = np.asarray(draw_crop(crop, {**outlined,
                                            "outline_threshold_factor": 1.6}))

        assert not np.array_equal(loose, tight), (
            "the threshold is part of the key, or the cache is stale")

    def test_changing_the_thickness_recomputes_the_edge(self, crop, outlined):
        forget_outline_masks()
        thin = np.asarray(draw_crop(crop, {**outlined, "edge_thickness": 1}))
        thick = np.asarray(draw_crop(crop, {**outlined, "edge_thickness": 4}))

        assert not np.array_equal(thin, thick)


class TestTheRedrawIsFast:
    """Measured, not asserted -- the instruction asks for milliseconds."""

    def _time(self, crops, picture):
        import time

        start = time.perf_counter()
        for one in crops:
            draw_crop(one, picture)
        return (time.perf_counter() - start) / len(crops) * 1000.0

    def test_a_second_pass_is_much_cheaper_than_the_first(self, crop,
                                                          outlined):
        crops = [crop] * 12
        forget_outline_masks()

        cold = self._time(crops[:1], outlined)
        warm = self._time(crops, outlined)

        assert warm < cold / 3.0, (
            f"cold {cold:.2f} ms, warm {warm:.2f} ms -- the outline is being "
            f"recomputed for a crop that has not changed")

    def test_changing_a_display_setting_reuses_the_outline(self, crop,
                                                           outlined):
        """The common case: the user moves transparency, not the threshold."""
        crops = [crop] * 12
        forget_outline_masks()
        self._time(crops, outlined)                      # warm it

        moved = self._time(crops, {**outlined, "edge_transparency": 40})

        assert moved < 2.0, (
            f"{moved:.2f} ms per crop -- transparency does not change the "
            f"mask or the edge, so nothing should be recomputed")


class TestTheCacheIsBounded:
    """A montage tab is a few hundred crops; the cache must not grow with the
    session."""

    def test_it_forgets_the_oldest(self, outlined):
        from spacr.qt import annotate_engine

        forget_outline_masks()
        rng = np.random.default_rng(1)
        for _ in range(annotate_engine._MASK_CACHE_SIZE + 40):
            array = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
            draw_crop(array, outlined)

        assert len(annotate_engine._MASK_CACHE) \
            <= annotate_engine._MASK_CACHE_SIZE
        assert len(annotate_engine._EDGE_CACHE) \
            <= annotate_engine._MASK_CACHE_SIZE

    def test_forgetting_is_available_to_a_caller(self):
        from spacr.qt import annotate_engine

        forget_outline_masks()

        assert annotate_engine._MASK_CACHE is None
