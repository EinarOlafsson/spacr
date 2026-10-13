"""Jitter and bar+jitter, the two graph types 200 named that had no kind.

    "I want to be able to choose between: Bar, Bar+jitter, Jitter, Box,
     Violin, Line (with and without spread), Scatter"

JITTER IS NOT SCATTER WITH NOISE ADDED. A scatter puts a point at its own x;
a jitter puts every point of a CATEGORY at that category's position and
spreads it sideways only so the points can be told apart. The displacement
carries no information and must not be readable as though it did.

BAR+JITTER IS THE MOST HONEST OF THE SUMMARY PLOTS, because the summary and
the thing summarised are in the same picture: a bar hiding four points and a
bar hiding four hundred look identical until the points are on it.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def canvas(qtbot):
    from PySide6.QtWidgets import QApplication

    from spacr.qt.widgets.graph_builder import GraphCanvas
    from spacr.qt.widgets.graph_spec import GraphSpec

    QApplication.instance() or QApplication([])
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "group": ["a"] * 60 + ["b"] * 60,
        "value": np.concatenate([rng.normal(10, 2, 60),
                                 rng.normal(13, 2, 60)])})

    def build(kind, **kwargs):
        widget = GraphCanvas()
        qtbot.addWidget(widget)
        widget.set_frame(frame)
        widget.set_spec(GraphSpec(x="group", y="value", kind=kind, **kwargs))
        return widget._figure.axes[0]
    return build


def _points(ax):
    return sum(len(c.get_offsets()) for c in ax.collections)


class TestTheKindsExist:

    def test_both_are_offered(self):
        from spacr.qt.widgets.graph_spec import (BAR_JITTER, JITTER,
                                                 PLOT_KINDS)

        assert JITTER in PLOT_KINDS
        assert BAR_JITTER in PLOT_KINDS


class TestJitterDrawsEveryObservation:
    """A summary plot hides the sample; this one is the sample."""

    def test_every_point_is_drawn(self, canvas):
        assert _points(canvas("jitter")) == 120

    def test_it_draws_no_bars(self, canvas):
        assert len(canvas("jitter").patches) == 0

    def test_the_axis_is_the_measurement_not_a_summary(self, canvas):
        assert canvas("jitter", spread="sd").get_ylabel() == "value"


class TestBarJitterShowsBoth:

    def test_the_bars_and_the_points_are_both_there(self, canvas):
        ax = canvas("bar_jitter")
        assert len(ax.patches) == 2
        assert _points(ax) >= 120

    def test_the_axis_states_the_whisker_like_a_bar(self, canvas):
        """It IS a bar; adding the points must not lose the label that says
        what the bar's whisker means."""
        assert "SD" in canvas("bar_jitter", spread="sd").get_ylabel()
        assert "SEM" in canvas("bar_jitter", spread="sem").get_ylabel()

    def test_the_bars_are_the_same_as_a_plain_bar(self, canvas):
        plain = [round(p.get_height(), 6) for p in canvas("bar").patches]
        both = [round(p.get_height(), 6)
                for p in canvas("bar_jitter").patches]
        assert plain == both


class TestTheDisplacementCarriesNoInformation:

    def test_the_points_of_a_level_sit_around_that_level(self, canvas):
        """Spread sideways only so they can be told apart -- never far
        enough to be read as a position."""
        ax = canvas("jitter")
        offsets = np.vstack([c.get_offsets() for c in ax.collections])
        for level in (0, 1):
            near = offsets[np.abs(offsets[:, 0] - level) < 0.5]
            assert len(near) == 60, level

    def test_it_is_seeded_so_a_redraw_is_identical(self, canvas):
        """A plot whose points move every repaint cannot be compared with
        the one in a slide from last week."""
        first = np.vstack([c.get_offsets()
                           for c in canvas("jitter", seed=7).collections])
        again = np.vstack([c.get_offsets()
                           for c in canvas("jitter", seed=7).collections])
        assert np.allclose(first, again)

    def test_a_different_seed_moves_them(self, canvas):
        first = np.vstack([c.get_offsets()
                           for c in canvas("jitter", seed=1).collections])
        other = np.vstack([c.get_offsets()
                           for c in canvas("jitter", seed=2).collections])
        assert not np.allclose(first, other)

    def test_the_values_themselves_are_untouched(self, canvas):
        """Only x is displaced. A jitter that moved the measurement would be
        drawing something that is not the data."""
        ax = canvas("jitter")
        offsets = np.vstack([c.get_offsets() for c in ax.collections])
        assert len(np.unique(np.round(offsets[:, 1], 9))) == 120
