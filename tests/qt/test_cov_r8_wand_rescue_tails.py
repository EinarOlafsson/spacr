"""The two wand rescues that give up quietly, and why they must.

Both are the same shape: the taper is asked to move a geometric edge onto
a real intensity one, and it comes back with nothing usable. What the
wand does then is keep what it already had -- a straight cut or a
geodesic arc is a worse answer than an intensity edge, and a much better
one than an empty region.

Neither arc is reachable by choosing an image: the taper answers with a
region for every input these tests could construct, which is why they
were still open after seven rounds. They are reached by making the taper
fail, which is the only thing that provokes them.

NOT COVERED HERE, and deliberately: `taper_region_to_intensity`'s own
`if not background.any() or not foreground.any()` bail-out. It cannot be
reached. `foreground` always contains the disc of radius two around the
seed intersected with `provisional`, and the function has already
returned unless `provisional[seed]` is True, so `foreground.any()` is
always True. `background` is `removed & (depth >= cutoff)` with
`cutoff = max(1.0, depth.max() * 0.75)`; an isolated pixel in `removed`
has a distance transform of 1.0, so the deepest pixel always clears its
own cutoff and `background.any()` is always True whenever `removed.any()`
is -- which the line above has already established. The guard is correct
defensive code for a function whose inputs could change; it is not a
missing test.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt import wand_rescue as W


def _leaky_image():
    """An object joined to a bright field by a one-pixel seam."""
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:31, 20:31] = 200.0                # the object
    image[25, 30:41] = 200.0                   # the seam it escapes along
    image[10:51, 40:61] = 200.0                # the field it escapes into
    return image


def test_a_taper_with_nothing_in_it_leaves_the_straight_cut_alone(
        monkeypatch):
    """The cut survives a taper that found no edge to move it to.

    `region` keeps whatever the directional trim and the intensity
    reflood agreed on, and `tapered` is NOT reported -- a report saying
    the edge was moved onto an intensity change, when it was not, would
    tell the user the wrong story about their own mask.
    """
    image = _leaky_image()

    called = []

    def finds_nothing(*args, **kwargs):
        called.append(1)
        return np.zeros(image.shape, dtype=bool)

    monkeypatch.setattr(W, "taper_region_to_intensity", finds_nothing)
    region, report = W.wand_region(image, 25, 25, 10.0,
                                   max_pixels=100_000,
                                   gradient_taper=True)

    assert called, "the taper was never reached, so this proves nothing"
    assert report["cuts"] == ["right"]
    assert not report.get("tapered"), "it claimed an edge it did not find"
    assert region.any(), "an empty taper emptied the region"
    assert region[25, 25], "the pixel the user clicked was thrown away"


def test_the_same_case_keeps_exactly_what_the_taper_was_offered(
        monkeypatch):
    """Not merely non-empty: unchanged.

    The region handed to the taper is the one that must come back, so
    this compares against the identical run with the taper switched off.
    """
    image = _leaky_image()
    expected, _ = W.wand_region(image, 25, 25, 10.0, max_pixels=100_000,
                                gradient_taper=False)

    monkeypatch.setattr(W, "taper_region_to_intensity",
                        lambda *a, **k: np.zeros(image.shape, dtype=bool))
    region, _report = W.wand_region(image, 25, 25, 10.0,
                                    max_pixels=100_000,
                                    gradient_taper=True)
    assert np.array_equal(region, expected)


class TestTheCappedArcWhenNoBandFits:
    """Over the budget, salvaged, and every band the taper tries fails.

    `taper_region_to_intensity` is called from TWO places: once on the
    directional cut, and again inside the cap loop. These fakes answer
    the first call with nothing, so the region reaching the cap is the
    large one, and only then vary what the loop is offered. Letting the
    first call succeed would shrink the region under the budget and the
    cap would never run -- which is what the first draft of these tests
    did, and why they failed.
    """

    @staticmethod
    def _fake(image, loop_answer):
        """Nothing for the cut; ``loop_answer(margin)`` for the cap loop."""
        state = {"calls": 0, "bands": []}

        def taper(_grey, _over, _bounded, _seed, sigma=None, margin=None,
                  foreground_erode=None):
            state["calls"] += 1
            if state["calls"] == 1:             # the directional-cut call
                return np.zeros(image.shape, dtype=bool)
            state["bands"].append(margin)
            return loop_answer(margin)

        return taper, state

    def test_the_geodesic_cap_is_kept_when_no_band_fits(self, monkeypatch):
        """Three bands are tried, and the arc is kept if none works.

        The loop narrows the band -- the margin, half of it, then one --
        and takes the first tapered result that is non-empty AND inside
        the budget. A taper that never returns one must leave the
        `cap_region_from_seed` answer standing.
        """
        image = _leaky_image()
        taper, state = self._fake(
            image, lambda _m: np.zeros(image.shape, dtype=bool))
        monkeypatch.setattr(W, "taper_region_to_intensity", taper)

        region, report = W.wand_region(image, 25, 25, 10.0,
                                       max_pixels=40,
                                       salvage_over_cap=True,
                                       gradient_taper=True,
                                       gradient_margin=8)

        assert report["capped"] is True
        assert not report.get("tapered")
        assert state["bands"] == [8, 4, 1], (
            "the band no longer narrows the way the comment says")
        assert 0 < int(region.sum()) <= 40, (
            "the salvaged region left the budget it was capped to")
        assert region[25, 25]

    def test_a_taper_over_the_budget_is_refused_at_every_band(self,
                                                             monkeypatch):
        """Non-empty is not enough -- it has to fit.

        A taper that answers with more pixels than the cap allows is the
        other way the loop runs out, and it must be rejected exactly as
        an empty one is.
        """
        image = _leaky_image()

        def too_big(_margin):
            out = np.zeros(image.shape, dtype=bool)
            out[10:51, 10:51] = True            # far over any small budget
            return out

        taper, state = self._fake(image, too_big)
        monkeypatch.setattr(W, "taper_region_to_intensity", taper)

        region, report = W.wand_region(image, 25, 25, 10.0,
                                       max_pixels=40,
                                       salvage_over_cap=True,
                                       gradient_taper=True)

        assert report["capped"] is True
        assert not report.get("tapered"), (
            "a taper bigger than the budget was accepted")
        assert state["bands"], "the cap loop never ran"
        assert int(region.sum()) <= 40

    def test_a_band_that_does_fit_is_taken_and_stops_the_search(self,
                                                               monkeypatch):
        """The positive side of the same loop, so the break is asserted.

        Without this the tests above would pass against a loop that never
        succeeds at all.
        """
        image = _leaky_image()

        def nine_pixels(_margin):
            out = np.zeros(image.shape, dtype=bool)
            out[24:27, 24:27] = True            # nine pixels, inside 40
            return out

        taper, state = self._fake(image, nine_pixels)
        monkeypatch.setattr(W, "taper_region_to_intensity", taper)

        region, report = W.wand_region(image, 25, 25, 10.0,
                                       max_pixels=40,
                                       salvage_over_cap=True,
                                       gradient_taper=True,
                                       gradient_margin=8)

        assert report["tapered"] is True
        assert report["capped"] is True
        assert state["bands"] == [8], (
            "it kept narrowing after a band had already fit")
        assert int(region.sum()) == 9
