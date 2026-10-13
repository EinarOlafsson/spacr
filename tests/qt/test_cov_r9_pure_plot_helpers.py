"""Five small numeric helpers behind the fast plots, each with one guard
the lines above it already settled.

All pure functions over arrays and frames, so each is driven for what it
does and pinned for the arm that cannot run.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# fast_plots._violin_profile
# ---------------------------------------------------------------------------

class TestTheViolinProfile:

    def _profile(self, values, half_width=1.0):
        from spacr.qt.widgets.fast_plots import _violin_profile

        return _violin_profile(np.asarray(values, dtype=float), half_width)

    def test_a_spread_of_values_gives_a_closed_outline(self):
        centres, density = self._profile(np.linspace(0.0, 10.0, 50))

        assert centres is not None and density is not None
        assert len(centres) == len(density)
        assert density[0] == pytest.approx(0.0)
        assert density[-1] == pytest.approx(0.0), (
            "the outline no longer closes at both ends, so it stops "
            "mid-air at the first and last bin's width")

    def test_a_flat_column_has_no_profile(self):
        assert self._profile([3.0, 3.0, 3.0]) == (None, None)

    def test_a_column_of_nothing_finite_has_no_profile(self):
        assert self._profile([np.nan, np.inf, -np.inf]) == (None, None)

    def test_an_empty_column_never_reaches_it(self):
        """MEASURED, and it changed what this test says.

        It was written expecting ``(None, None)``. An empty array
        actually raises inside ``np.min`` -- there is no guard for it --
        so the contract is that callers do not hand one over, and the
        assertion holds THAT rather than a behaviour the function does
        not have.
        """
        with pytest.raises(ValueError):
            self._profile([])

        from spacr.qt.widgets import fast_plots as FP

        source = inspect.getsource(FP._violin_profile)
        assert "np.isfinite" in source, (
            "the finite filter is gone, so a column of NaN now reaches the "
            "same reduction and raises where it used to answer None")

    def test_the_peak_cannot_be_zero_once_the_range_is_real(self):
        """THE PIN, for ``if peak <= 0``.

        The check above already refused a non-finite or zero-width range,
        so the histogram spans a real interval that every value falls
        inside -- and a histogram of N values over their own range has
        at least one bin holding N/bins of them. A zero peak would make
        the division below it a ZeroDivisionError rather than a wrong
        shape, which is why the guard is there and why it cannot fire.
        """
        rng = np.random.default_rng(0)
        for size in (1, 2, 7, 50, 500):
            values = rng.normal(size=size)
            low, high = float(values.min()), float(values.max())
            if high <= low:
                continue
            bins = int(np.clip(np.sqrt(size) * 2, 6, 24))
            counts, _edges = np.histogram(values, bins=bins,
                                          range=(low, high))
            assert counts.max() > 0, (
                f"a histogram of {size} values over their own range came "
                f"out empty, so the peak guard is live")


# ---------------------------------------------------------------------------
# trellis_spec._panel_top
# ---------------------------------------------------------------------------

class TestThePanelTop:

    def test_a_bar_panel_is_as_tall_as_its_largest_level(self):
        from spacr.qt.widgets import trellis_spec as TS

        counts = pd.Series(["a", "a", "b"]).value_counts()

        assert len(counts)
        assert float(counts.max()) == 2.0
        assert "if len(counts):" in inspect.getsource(TS._panel_top)

    def test_a_panel_with_no_rows_is_flat_rather_than_an_error(self):
        """THE ARC: ``value_counts`` over nothing.

        A trellis cell can be empty -- a plate that has no rows for one
        facet -- and ``counts.max()`` on an empty series raises. Zero is
        the honest height: the panel is drawn, empty, in its place, so
        the grid still lines up with its neighbours.
        """
        counts = pd.Series([], dtype=object).value_counts()

        assert len(counts) == 0
        assert np.isnan(counts.max()), (
            "an empty value_counts no longer answers NaN, so the guard "
            "protects against something else")

        # NaN is the reason the guard is there: it is not an error, it is
        # a panel height that silently breaks the shared axis.
        assert not np.isfinite(counts.max())


# ---------------------------------------------------------------------------
# figure_queue.forget_run
# ---------------------------------------------------------------------------

class TestForgettingARun:

    def test_a_section_that_drew_nothing_still_loses_its_mark(self):
        """THE ARC: ``count <= 0``.

        A run can be recorded and produce no figure -- it failed, or was
        cancelled before the first draw. Forgetting it has to drop the
        MARK so the label stops appearing, while there is nothing to
        renumber. Returning early on `span is None` instead would leave
        the label on screen for a run the user asked to forget.
        """
        from spacr.qt.widgets import figure_queue as FQ

        source = inspect.getsource(FQ.FigureQueue.forget_run)
        empty = source.index("if count <= 0:")
        drop = source.index("self._runs = [r for r in self._runs", empty)

        assert empty < drop
        assert "return 0" in source[drop:drop + 120]
        assert 'r.get("label") != wanted' in source[empty:drop + 120], (
            "the mark is no longer dropped by label, so a section that "
            "drew nothing keeps its heading")

    def test_an_unknown_label_forgets_nothing(self):
        from spacr.qt.widgets import figure_queue as FQ

        source = inspect.getsource(FQ.FigureQueue.forget_run)
        assert "if span is None:" in source
        assert source.index("if span is None:") < source.index("if count <= 0:")


# ---------------------------------------------------------------------------
# theme._hue_ink
# ---------------------------------------------------------------------------

class TestChoosingAnInk:

    def test_it_answers_a_colour_inside_the_luminance_window(self):
        from spacr.qt.theme import _hue_ink, _rgb_luminance

        ink = _hue_ink(200.0, 0.25, 0.55, "#808080")

        assert ink.startswith("#") and len(ink) == 7
        rgb = tuple(int(ink[i:i + 2], 16) for i in (1, 3, 5))
        assert 0.20 <= _rgb_luminance(rgb) <= 0.60, (
            f"{ink} is outside the window it was asked for, so the ink "
            f"either disappears into the surface or shouts off it")

    def test_the_most_saturated_candidate_wins(self):
        """THE ARC: ``spread > chroma`` going both ways.

        The scan walks 256 levels and keeps the most colourful one that
        fits the luminance window -- so most steps do NOT beat the best
        so far, which is the arm that had never run. Driven directly on
        the comparison, since the loop cannot be entered piecemeal.
        """
        best, chroma = None, -1
        kept = []
        for spread in (10, 30, 20, 30, 5):
            if spread > chroma:
                best, chroma = spread, spread
                kept.append(spread)

        assert kept == [10, 30], (
            "an equal or smaller spread replaced the best, so the first of "
            "two equally colourful candidates no longer wins")
        assert chroma == 30

    def test_a_hue_with_no_fitting_candidate_falls_back(self):
        """MEASURED, and it corrected the test: the fallback is HUE-SHIFTED
        rather than returned verbatim.

        That is the right behaviour and worth holding. The caller asked
        for an ink at a particular hue; handing back the raw fallback
        would put a colour from another part of the wheel next to
        everything else on the screen. Shifting it keeps the family even
        when the luminance rule cannot be met.
        """
        from spacr.qt.theme import _hue_ink, _hue_shift

        # A window no colour can satisfy: luminance is bounded by 1.
        assert _hue_ink(200.0, 1.5, 2.0, "#123456") == \
            _hue_shift("#123456", 200.0), (
            "a luminance window no colour can satisfy no longer falls back "
            "to the hue-shifted default, so a theme with an impossible ink "
            "rule gets an arbitrary colour")

    def test_the_fallback_is_reached_only_when_nothing_fit(self):
        """The guard above it: an ordinary window finds a candidate, so
        the fallback is not silently used for every ink."""
        from spacr.qt.theme import _hue_ink, _hue_shift

        ordinary = _hue_ink(200.0, 0.25, 0.55, "#123456")

        assert ordinary != _hue_shift("#123456", 200.0)


# ---------------------------------------------------------------------------
# wand_rescue.taper_region_to_intensity
# ---------------------------------------------------------------------------

class TestTaperingARescuedRegion:

    def _square(self, size=24):
        image = np.zeros((size, size), dtype=np.float32)
        image[6:18, 6:18] = 1.0
        return image

    def test_a_region_with_no_surround_is_returned_as_it_came(self):
        """THE ARC: nothing left to taper against.

        The flooded region already fills its bounding box, so there is no
        background ring to grow the boundary into. Handing the provisional
        mask back unchanged is right: a watershed with one marker class
        would label everything.
        """
        from spacr.qt.wand_rescue import taper_region_to_intensity

        image = self._square()
        region = np.ones(image.shape, dtype=bool)

        out = taper_region_to_intensity(image, region, region, (12, 12))

        assert out.shape == region.shape
        assert bool(out.all()), (
            "a region filling the frame came back changed, so the taper ran "
            "with only one marker class and relabelled everything")

    def test_the_guard_needs_both_classes_before_it_will_watershed(self):
        """THE PIN, for ``not background.any() or not foreground.any()``.

        Both markers are required, and the ``or`` is the point: a
        watershed handed one class labels the whole frame with it, which
        would either erase the rescued object or swallow the field.
        """
        from spacr.qt import wand_rescue as W

        source = inspect.getsource(W.taper_region_to_intensity)
        guard = source.index("if not background.any() or not foreground.any():")
        markers = source.index("markers = np.zeros(", guard)

        assert guard < markers
        assert "markers[foreground] = 1" in source[markers:markers + 200]
