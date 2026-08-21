"""A bar's whisker is a choice, and the plot says which one was made.

Instruction 204: "for the cell table graphs if bar is chosen the user should
be able to choose SD, Var, or SEM error bars."

THEY ARE NOT INTERCHANGEABLE. SD describes the CELLS and does not shrink as
you measure more of them; SEM describes the MEAN and shrinks as sqrt(n), so
it is small whenever n is large whatever the biology does. At n=3000 they
differ fifty-five-fold. A reader who assumes the wrong one reads a real
effect as noise, or noise as a real effect -- so the answer goes on the
axis, not only in a settings dialog the reader never opens.
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.figures.spread import (SPREAD_CHOICES, SPREAD_NONE, SPREAD_SD,
                                  SPREAD_SEM, SPREAD_VAR, spread_label,
                                  spread_of, summarise)


@pytest.fixture
def sample():
    return np.random.default_rng(0).normal(10.0, 2.0, 3000)


class TestTheArithmetic:

    def test_sd_is_the_sample_standard_deviation(self, sample):
        assert spread_of(sample, SPREAD_SD) == pytest.approx(
            float(np.std(sample, ddof=1)))

    def test_sem_is_sd_over_root_n(self, sample):
        assert spread_of(sample, SPREAD_SEM) == pytest.approx(
            spread_of(sample, SPREAD_SD) / np.sqrt(sample.size))

    def test_var_is_sd_squared(self, sample):
        assert spread_of(sample, SPREAD_VAR) == pytest.approx(
            spread_of(sample, SPREAD_SD) ** 2)

    def test_the_two_that_get_confused_differ_by_root_n(self, sample):
        ratio = spread_of(sample, SPREAD_SD) / spread_of(sample, SPREAD_SEM)
        assert ratio == pytest.approx(np.sqrt(sample.size), rel=1e-6)
        assert ratio > 50, "the confusion is not academic at this n"

    def test_ddof_one_because_these_are_a_sample(self):
        """The population formula understates the spread by sqrt((n-1)/n) --
        ten per cent at n=5, which is exactly the well that needed the
        whisker."""
        small = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert spread_of(small, SPREAD_SD) > float(np.std(small))


class TestOneObservationHasNoSpread:
    """A zero-length whisker says "no variation measured" where the truth is
    "not measurable", and those look identical on a plot."""

    @pytest.mark.parametrize("kind", [SPREAD_SD, SPREAD_SEM, SPREAD_VAR])
    def test_it_is_nan_not_zero(self, kind):
        assert np.isnan(spread_of([7.0], kind))

    def test_an_empty_group_is_nan_too(self):
        assert np.isnan(spread_of([], SPREAD_SD))

    def test_non_finite_values_are_dropped_before_counting(self):
        assert np.isnan(spread_of([1.0, np.nan], SPREAD_SD))


class TestTheLabel:

    @pytest.mark.parametrize("kind,expected", [
        (SPREAD_SD, "SD"), (SPREAD_SEM, "SEM"), (SPREAD_VAR, "variance"),
    ])
    def test_it_names_the_quantity(self, kind, expected):
        assert expected in spread_label(kind)

    def test_variance_squares_the_unit(self):
        """A variance whisker on an axis labelled with the plain unit is
        wrong in a way that looks fine."""
        assert "²" in spread_label(SPREAD_VAR, unit="px")
        assert "²" not in spread_label(SPREAD_SD, unit="px")

    def test_none_has_no_label(self):
        assert spread_label(SPREAD_NONE) == ""

    def test_an_unknown_spread_is_refused(self):
        with pytest.raises(ValueError):
            spread_of([1.0, 2.0], "stderr")
        with pytest.raises(ValueError):
            spread_label("stderr")


class TestSummarise:

    def test_it_reports_mean_spread_and_n(self):
        out = summarise({"a": [1.0, 2.0, 3.0]}, SPREAD_SD)
        assert out["a"]["mean"] == pytest.approx(2.0)
        assert out["a"]["n"] == 3.0

    def test_an_empty_level_is_omitted_not_drawn_at_zero(self):
        """An empty bar and a bar of zeros are different claims."""
        out = summarise({"a": [1.0, 2.0], "b": []}, SPREAD_SD)
        assert "b" not in out

    def test_a_level_of_all_nan_is_omitted(self):
        out = summarise({"a": [np.nan, np.nan]}, SPREAD_SD)
        assert out == {}


class TestTheChoicesAreOffered:

    def test_all_four_are_listed_with_what_they_describe(self):
        values = [value for value, _label in SPREAD_CHOICES]
        assert values == [SPREAD_NONE, SPREAD_SD, SPREAD_SEM, SPREAD_VAR]
        labels = " ".join(label for _v, label in SPREAD_CHOICES)
        # The label says what the quantity DESCRIBES, since that is the
        # thing a reader gets wrong.
        assert "cells" in labels
        assert "mean" in labels


class TestItReachesThePlot:

    @pytest.fixture
    def canvas(self):
        pytest.importorskip("PySide6")
        import matplotlib
        matplotlib.use("Agg")
        import pandas as pd
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets.graph_builder import GraphCanvas
        from spacr.qt.widgets.graph_spec import GraphSpec

        QApplication.instance() or QApplication([])
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            "group": ["a"] * 300 + ["b"] * 300,
            "value": np.concatenate([rng.normal(10, 2, 300),
                                     rng.normal(12, 2, 300)])})

        def build(spread):
            widget = GraphCanvas()
            widget.set_frame(frame)
            widget.set_spec(GraphSpec(x="group", y="value", kind="bar",
                                      spread=spread))
            return widget._figure.axes[0]
        return build

    def test_the_bar_is_the_mean_not_the_count(self, canvas):
        ax = canvas(SPREAD_NONE)
        heights = sorted(round(p.get_height(), 1) for p in ax.patches)
        assert heights == [pytest.approx(9.9, abs=0.3),
                           pytest.approx(12.0, abs=0.3)]

    @pytest.mark.parametrize("kind,expected", [
        (SPREAD_SD, "SD"), (SPREAD_SEM, "SEM"), (SPREAD_VAR, "variance"),
    ])
    def test_the_axis_states_which_whisker(self, canvas, kind, expected):
        assert expected in canvas(kind).get_ylabel()

    def test_no_whisker_leaves_the_plain_column_name(self, canvas):
        assert canvas(SPREAD_NONE).get_ylabel() == "value"

    def test_a_count_bar_is_still_a_count_bar(self, canvas):
        """With nothing numeric to average there is nothing to be spread
        about, and an error bar would be a statement about a number that is
        exact."""
        import pandas as pd
        from PySide6.QtWidgets import QApplication

        from spacr.qt.widgets.graph_builder import GraphCanvas
        from spacr.qt.widgets.graph_spec import GraphSpec

        QApplication.instance() or QApplication([])
        widget = GraphCanvas()
        widget.set_frame(pd.DataFrame({"group": ["a"] * 3 + ["b"] * 5}))
        widget.set_spec(GraphSpec(x="group", kind="bar", spread=SPREAD_SD))
        assert widget._figure.axes[0].get_ylabel() == "count"
