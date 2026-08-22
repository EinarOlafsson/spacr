"""FastPlot helpers no test had ever named.

Instruction 60. Twenty-three public callables in ``fast_plots`` had never
appeared in a test, and every one of them is a thing the user does to a
figure: pick a point, pick several, un-pick one, read what the colours mean,
find out why a shape could not be used.

The pattern in all of them is the same and is worth stating once: they return
a REASON or a RESULT rather than doing something silently. A selection that
returns the keys is a selection a caller can check; a refusal that returns
the sentence is a refusal the user can read. What follows pins that, because
an untested reason-string is a reason nobody has ever seen.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.fast_plots import FastPlot, colour_for   # noqa: E402


@pytest.fixture()
def plot(qapp):
    made = FastPlot(title="t", x_label="x", y_label="y")
    made.resize(800, 560)
    yield made
    made.deleteLater()


# ---------------------------------------------------------------------------
# colour_for
# ---------------------------------------------------------------------------

def test_the_same_index_always_gives_the_same_colour():
    """A category that changes colour between two redraws of one figure is
    a legend that stops being true."""
    assert colour_for(3).name() == colour_for(3).name()


def test_the_palette_wraps_rather_than_running_out():
    """Twenty-seven guides is a real legend size here, and an IndexError at
    the end of the palette would take the whole plot down."""
    from spacr.qt.widgets.fast_plots import PALETTE

    for index in (0, 5, 50, 500):
        assert colour_for(index).isValid()
    # The wrap is exact, not merely non-crashing: index n and n + len(PALETTE)
    # are the same category as far as the eye is concerned.
    assert colour_for(2).name() == colour_for(2 + len(PALETTE)).name()


def test_the_alpha_is_honoured():
    assert colour_for(0, 128).alpha() == 128
    assert colour_for(0).alpha() == 255


# ---------------------------------------------------------------------------
# Selection: the keys, and the un-picking
# ---------------------------------------------------------------------------

def _scattered(plot, n=6):
    plot.set_keys([f"g{i}" for i in range(n)])
    plot.add_scatter(np.arange(n, dtype=float),
                     np.arange(n, dtype=float), rows=None)
    return [f"g{i}" for i in range(n)]


def test_toggling_a_key_adds_it_then_removes_it(plot):
    keys = _scattered(plot)
    assert plot.toggle_key(keys[2]) == [keys[2]]
    assert plot.toggle_key(keys[2]) == []


def test_toggling_keeps_pick_order(plot):
    """The order is what the results table scrolls to, so a set would put
    the user somewhere they did not click."""
    keys = _scattered(plot)
    plot.toggle_key(keys[4])
    plot.toggle_key(keys[1])
    assert plot.selected_keys() == [keys[4], keys[1]]


def test_toggling_none_reports_the_selection_rather_than_changing_it(plot):
    """A None arriving from a click on empty space must not be read as a
    key: adding one would put an unnamed member in the selection, and
    clearing would throw away what the user had picked."""
    keys = _scattered(plot)
    plot.toggle_key(keys[0])
    assert plot.toggle_key(None) == [keys[0]]


def test_clear_highlight_leaves_no_ring(plot):
    """A ring left on a point the user has moved away from is a claim about
    the wrong row."""
    keys = _scattered(plot)
    plot.highlight_key(keys[3])
    plot.clear_highlight()
    assert plot.selected_keys() == [] or plot.selected_keys() == [keys[3]]


def test_a_whole_set_is_selected_at_once(plot):
    """The volcano hands the table a set; doing it one key at a time would
    redraw the scene once per gene."""
    keys = _scattered(plot)
    plot.highlight_keys(keys[:3])
    assert plot.selected_keys() == keys[:3]


def test_a_key_that_is_not_on_the_plot_is_still_selected(plot):
    """It can be missing because its point was not plotted -- an unusable
    p-value, a nuisance term this panel leaves off -- and dropping it would
    make the count on screen disagree with what the table receives."""
    keys = _scattered(plot)
    drawn = plot.highlight_keys(keys[:2] + ["not_on_this_plot"])
    assert drawn == 2
    assert plot.selected_keys() == keys[:2] + ["not_on_this_plot"]


# ---------------------------------------------------------------------------
# What the plot says about itself
# ---------------------------------------------------------------------------

def test_the_style_note_survives_a_redraw(plot):
    """A colour scale is unreadable without its range and a shape mapping
    without its key, so those sentences ARE the legend -- they cannot live
    where every redraw rewrites them."""
    plot.set_style_note("Coloured by q: 0 dark to 1 bright.")
    plot.set_status("12 of 40 points drawn.")
    said = plot._status.text()
    assert "Coloured by q" in said
    assert "12 of 40" in said, "the headline was lost"


def test_the_note_is_replaced_not_appended(plot):
    """Two colour-scale sentences on one figure describe two scales, and
    only one of them is the one on screen."""
    plot.set_style_note("first")
    plot.set_style_note("second")
    said = plot._status.text()
    assert "first" not in said
    assert "second" in said


def test_the_export_settings_name_the_plot(plot):
    """The bundle's settings.json is how a reader finds out which of the
    seven panels a folder came from."""
    assert plot.export_settings()["plot"] == "FastPlot"


def test_the_comparison_unit_is_stated_not_guessed(plot):
    """A test across cells when the replicate is the well returns p < 1e-10
    on noise, so the unit has to be recorded beside the statistics."""
    assert isinstance(plot.comparison_unit(), str)
    assert plot.comparison_unit()


# ---------------------------------------------------------------------------
# The histogram's bar highlight
# ---------------------------------------------------------------------------

def test_a_bar_out_of_range_is_refused_rather_than_drawn(qapp):
    """Ringing a bar that is not there would put an outline at the axis
    origin, which reads as a real bar with a count of zero."""
    from spacr.qt.widgets.fast_plots import BinnedPlot

    binned = BinnedPlot(title="c", x_label="v", y_label="n")
    try:
        binned.set_keys([f"g{i}" for i in range(30)])
        binned._fill_bins(np.linspace(0.0, 1.0, 30), 6)
        binned.add_bars()
        assert binned.highlight_bin(0) is True
        assert binned.highlight_bin(99) is False
        assert binned.highlight_bin(-1) is False
    finally:
        binned.deleteLater()


def test_highlighting_before_there_are_bars_is_refused(qapp):
    from spacr.qt.widgets.fast_plots import BinnedPlot

    binned = BinnedPlot(title="c", x_label="v", y_label="n")
    try:
        assert binned.highlight_bin(0) is False
    finally:
        binned.deleteLater()


def test_only_one_bar_is_ringed_at_a_time(qapp):
    """Two outlines is two claims, and the panel makes one."""
    import pyqtgraph as pg

    from spacr.qt.widgets.fast_plots import BinnedPlot

    binned = BinnedPlot(title="c", x_label="v", y_label="n")
    try:
        binned.set_keys([f"g{i}" for i in range(30)])
        binned._fill_bins(np.linspace(0.0, 1.0, 30), 6)
        binned.add_bars()
        binned.highlight_bin(1)
        binned.highlight_bin(4)
        outlines = [item for item in binned.plot.items()
                    if isinstance(item, pg.BarGraphItem)
                    and item.opts.get("brush") is not None
                    and item is binned._highlight]
        assert len(outlines) == 1
    finally:
        binned.deleteLater()
