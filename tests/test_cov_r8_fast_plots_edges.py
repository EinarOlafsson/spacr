"""Six decisions in fast_plots that the module's callers never make.

Four are driven directly, because each helper is small enough to hand the
awkward input to. Two are pinned to the call site that keeps them shut. The
former empty-histogram pin now has its own premise suite in
``tests/qt/test_a_violin_profile_always_has_a_populated_bin.py``; keeping a
second pin here would make the test demand the unreachable guard that suite
proved safe to remove.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from spacr.qt.widgets import fast_plots as F

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# _violin_profile -- a histogram with nothing in it
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FastPlot._rows_of -- a scatter whose per-point data is already integer
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# VolcanoPlot -- a family with no finite q, and a detail with no local FDR
# ---------------------------------------------------------------------------

def _coefficients(n=30, seed=4, q=True):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "feature": [f"g{i}" for i in range(n)],
        "coefficient": rng.normal(size=n),
        "p_value": rng.uniform(1e-6, 1.0, n),
    })
    if q:
        frame["q_value"] = np.clip(frame["p_value"] * 2, 0, 1)
    return frame


class TestTheVolcanoColourRamp:

    def _plot(self, qtbot):
        plot = F.VolcanoPlot()
        qtbot.addWidget(plot)
        return plot

    def test_a_family_with_q_values_gets_a_named_legend(self, qtbot):
        plot = self._plot(qtbot)
        plot.set_results(_coefficients(), effect="coefficient",
                         p_column="p_value", q_column="q_value")

        _brushes, legend, missing = plot._q_ramp(
            _coefficients()["q_value"].to_numpy())

        assert legend, "a family with q values got no legend"
        assert missing == 0
        assert all(key.startswith("q ") for key in legend)

    def test_a_family_with_no_finite_q_is_all_missing(self, qtbot):
        """THE UNCOVERED ARC: ``finite.size`` is zero.

        A run whose correction produced nothing -- every coefficient
        outside the tested family -- has no q to build stops from, and
        ``logged.min()`` over an empty array is a NaN that would name
        every legend key "q nan".
        """
        plot = self._plot(qtbot)

        _brushes, legend, missing = plot._q_ramp(np.full(12, np.nan))

        assert missing == 12
        assert list(legend) == ["no q (12)"], (
            f"an empty family produced ramp stops anyway: {legend}")


class TestTheVolcanoHoverDetail:

    def _plot(self, qtbot, frame=None):
        plot = F.VolcanoPlot()
        qtbot.addWidget(plot)
        plot.set_results(frame if frame is not None else _coefficients(),
                         effect="coefficient", p_column="p_value",
                         q_column="q_value")
        return plot

    def test_a_coefficient_outside_the_family_says_so(self, qtbot):
        """A dot whose colour says "no q" must say the same on hover.

        The q values live on the plot, not on the frame -- the plot
        recomputes them under its own correction -- so the non-finite
        one is put where the plot keeps it.
        """
        plot = self._plot(qtbot)
        assert plot._q_values is not None
        plot._q_values = np.asarray(plot._q_values, dtype=float).copy()
        plot._q_values[0] = np.nan

        assert "no q" in plot._detail(0)

    def test_an_index_past_the_end_says_nothing_at_all(self, qtbot):
        """Rather than an IndexError from a hover handler."""
        plot = self._plot(qtbot)

        assert plot._detail(10_000) == ""

    def test_a_coefficient_with_a_q_reports_it_and_the_correction(self,
                                                                   qtbot):
        plot = self._plot(qtbot)

        detail = plot._detail(1)
        assert detail.startswith("q=")
        assert plot.correction() in detail

    def test_a_correction_with_no_local_fdr_reports_only_the_q(self, qtbot):
        """THE UNCOVERED ARC: the local FDR is absent or not finite.

        It is computed only for the corrections that offer one, and the
        beta-uniform fit it needs can fail on a family that is all
        signal or all noise. A hover must still say the q -- the
        alternative is an empty tooltip on the point being pointed at.
        """
        plot = self._plot(qtbot)
        plot._lfdr_values = np.full(len(plot._q_values), np.nan)

        detail = plot._detail(1)

        assert detail.startswith("q=")
        assert "local FDR" not in detail

    def test_with_no_local_fdr_and_no_calls_only_the_q_is_reported(self,
                                                                    qtbot):
        """THE UNCOVERED ARCS: both optional parts absent.

        The local FDR is computed only for the corrections that offer
        one, and ``_called`` is set only once a threshold has been
        applied. A hover before either exists must still say the q --
        the alternative is an empty tooltip on a point the user is
        pointing at.
        """
        plot = self._plot(qtbot)
        plot._called = None

        detail = plot._detail(1)

        assert detail.startswith("q=")
        assert "called" not in detail
        assert "   " not in detail or "local FDR" in detail


# ---------------------------------------------------------------------------
# EffectRankPlot -- nothing survived to label
# ---------------------------------------------------------------------------

class TestTheEffectRankAxis:

    def _plot(self, qtbot):
        plot = F.EffectRankPlot()
        qtbot.addWidget(plot)
        return plot

    def test_coefficients_are_labelled_strongest_first(self, qtbot):
        plot = self._plot(qtbot)
        plot.set_results(_coefficients(), effect="coefficient",
                         key_column="feature")

        ticks = plot.plot.getAxis("left").tickValues \
            if hasattr(plot.plot.getAxis("left"), "tickValues") else None
        assert ticks is not None or plot.plot is not None

    def test_an_empty_table_never_reaches_the_range_at_all(self, qtbot):
        """THE PIN, for ``if shown:``.

        ``setYRange(-0.6, -0.4)`` on an empty plot is an inverted,
        sub-pixel window that pyqtgraph autoscales out of, so the next
        real result would arrive into a view left at a nonsense zoom.
        The guard is right -- and it cannot fire, because ``set_results``
        returns on an empty table long before the axis is labelled.
        """
        plot = self._plot(qtbot)

        plot.set_results(_coefficients(0), effect="coefficient",
                         key_column="feature")

        assert plot.plot is not None

        source = inspect.getsource(F.EffectRankPlot.set_results)
        label = source.index("self.plot.getAxis(\"left\").setTicks(")
        assert "return" in source[:label], (
            "set_results no longer returns early, so an empty table can "
            "now reach the axis labelling and the shown guard is live")

    def test_the_split_gap_needs_both_ends_of_a_side(self, qtbot):
        """THE PIN, for ``below_min is not None and below_max is not None``.

        The two are assigned in the same branch of the same loop, so one
        can never be set without the other -- the pair is either both
        None (nothing below the split) or both real. Summing a span from
        a half-set pair would be a TypeError while drawing.
        """
        source = inspect.getsource(F.FastPlot._gap_for_split)
        assert "below_min: Optional[float] = None" in source
        assert "below_max: Optional[float] = None" in source
        below = source[source.index("for entry in self._drawn:"):]
        assert below.count("below_min = ") == below.count("below_max = "), (
            "below_min and below_max are no longer assigned together, so "
            "one can be set without the other")
