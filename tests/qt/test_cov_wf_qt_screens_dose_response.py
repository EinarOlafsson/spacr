"""The Dose–Response screen's four remaining un-driven decisions.

``tests/qt/test_dose_response_screen.py`` fits a synthetic plate and reads the
engine's numbers back out of the grid; ``test_cov_w3_7_dose_response.py``
drives the loading half. What is left are four small forks that nothing in
either file takes, and all four are places where the screen has to *not* do
something:

* the concentration picker walking past a numeric column whose name does not
  look like a dose before it lands on the one that does;
* the figure drawn with no row selected -- every curve, no EC50 marker;
* a bounded EC50 whose interval has an open end -- the line, but no band;
* closing a screen whose canvas has no pending-draw timer to cancel.

Each is exercised here beside the input that takes the other side of the same
fork, so an assertion that something is absent is always paired with the
input that produces it.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.screens.dose_response import DoseResponseScreen
from spacr.qt.widgets.dose_response import four_parameter_logistic

pytestmark = pytest.mark.qt

#: An eight-point three-fold dilution series, 27 µM down to about 12 nM.
DOSES = 27.0 / 3.0 ** np.arange(8)


def _series(ec50: float, seed: int):
    """A clean inhibition curve through ``DOSES``, in triplicate."""
    dose = np.repeat(DOSES, 3)
    clean = four_parameter_logistic(dose, 0.0, 100.0, np.log10(ec50), -1.0)
    rng = np.random.default_rng(seed)
    return dose, clean + rng.normal(0.0, 1.0, dose.size)


@pytest.fixture()
def two_gene_frame() -> pd.DataFrame:
    """Two compounds whose midpoints the tested range brackets."""
    parts = []
    for gene, ec50, seed in (("geneA", 1.0, 1), ("geneB", 0.3, 2)):
        dose, response = _series(ec50, seed)
        parts.append(pd.DataFrame({"gene": gene, "conc_uM": dose,
                                   "signal": response}))
    return pd.concat(parts, ignore_index=True)


@pytest.fixture()
def fitted(qtbot, two_gene_frame):
    """A screen that has already fitted both curves, row 0 selected."""
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(two_gene_frame, label="synthetic")
    widget.concentration_picker.setCurrentText("conc_uM")
    widget.response_picker.setCurrentText("signal")
    widget.group_picker.setCurrentText("gene")
    widget.unit_edit.setText("µM")
    widget.fit()
    assert widget.result_set() is not None, "the inline fit produced nothing"
    return widget


def _axes(widget):
    """The one subplot ``_draw`` builds, whatever it last drew."""
    return widget._figure.axes[0]


def _dashed(axes):
    """The EC50 lines: dashed for a bounded midpoint, dotted for a bound."""
    return [line for line in axes.lines if line.get_linestyle() == "--"]


# ---------------------------------------------------------------------------
# Guessing the dose column
# ---------------------------------------------------------------------------

def test_the_dose_picker_walks_past_a_numeric_column_that_is_not_a_dose(
        qtbot):
    """A plate carries more than one column of positive numbers.

    ``assay_level`` sorts before ``conc_uM`` and is just as fittable — eight
    distinct positive values — so it is what the picker would land on if the
    initial guess were "the first candidate". The guess is by *name*, and it
    has to keep walking the list until a name says dose. Landing on the wrong
    column means the first Fit a user presses draws a curve through an axis
    that is not a concentration, and the EC50 it prints is a number about
    nothing.
    """
    dose, response = _series(1.0, seed=7)
    frame = pd.DataFrame({"assay_level": dose * 2.0, "conc_uM": dose,
                          "signal": response})
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.set_frame(frame, label="two numeric columns")

    offered = [widget.concentration_picker.itemText(i)
               for i in range(widget.concentration_picker.count())]
    assert offered[0] == "assay_level", (
        "the column that has to be walked past is not even in the list")
    assert "conc_uM" in offered
    assert widget.concentration_picker.currentText() == "conc_uM"
    assert widget.fit_button.isEnabled() is True


def test_a_table_with_no_dose_shaped_name_leaves_the_first_column_showing(
        qtbot):
    """When no name says dose, the screen does not invent a preference.

    The initial guess is a convenience and never a classifier: a table whose
    columns are ``x`` and ``y`` gets the combo box's own first entry, not a
    column silently promoted because it happened to be numeric. The user is
    still the one who says which column is the dose, and the previous test's
    plate is what shows the guess firing when a name does earn it.
    """
    dose, response = _series(1.0, seed=11)
    frame = pd.DataFrame({"x": dose, "y": response})
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)

    widget.set_frame(frame, label="anonymous columns")

    offered = [widget.concentration_picker.itemText(i)
               for i in range(widget.concentration_picker.count())]
    assert offered[:2] == ["x", "y"]
    assert widget.concentration_picker.currentText() == "x"
    assert widget.report.toPlainText() == ""


# ---------------------------------------------------------------------------
# The figure with and without a selected row
# ---------------------------------------------------------------------------

def test_a_figure_with_no_row_selected_draws_every_curve_and_no_ec50_mark(
        fitted):
    """The EC50 marker belongs to the selected row, not to the plate.

    A 96-compound plate with a vertical line and a shaded band per compound
    is not a figure, so the marker is drawn for the chosen row only — and the
    unchosen state has to be a real state, because that is what the screen
    shows between a fit finishing and a row being clicked. Both halves are
    driven here: with no selection the picture is points and curves only,
    with row 0 selected the dashed line and its annotation appear.
    """
    fitted._draw(None)
    axes = _axes(fitted)
    unselected_lines = len(axes.lines)

    assert unselected_lines == 4, "two curves, each a marker line and a fit"
    assert _dashed(axes) == []
    assert [text.get_text() for text in axes.texts] == []
    assert len(axes.patches) == 0
    # Nothing is faded, because nothing is out of focus.
    assert {line.get_alpha() for line in axes.lines} == {0.9, 1.0}
    assert axes.get_xscale() == "log"

    fitted._draw(0)
    axes = _axes(fitted)

    assert len(_dashed(axes)) == 1
    assert len(axes.lines) == unselected_lines + 1
    marked = [text.get_text() for text in axes.texts]
    assert len(marked) == 1 and marked[0].startswith("EC50 ")
    assert 0.25 in {line.get_alpha() for line in axes.lines}, (
        "the unselected curve should have faded back")


def test_the_axis_limits_belong_to_the_measurements_not_to_the_interval(
        fitted):
    """A twenty-decade interval must not shrink the data to one pixel.

    The EC50 band is drawn *after* the x range is read and the range is put
    back afterwards, so a poorly determined midpoint cannot rescale the axis
    away from the doses that were actually tested. Driving the same figure
    with and without the marker is what shows the range surviving.
    """
    fitted._draw(None)
    plain = _axes(fitted).get_xlim()

    fitted._draw(0)
    marked = _axes(fitted).get_xlim()

    assert marked == pytest.approx(plain)
    assert marked[0] < float(DOSES.min())
    assert marked[1] > float(DOSES.max())


# ---------------------------------------------------------------------------
# A bounded EC50 whose interval has an open end
# ---------------------------------------------------------------------------

def _with_first_result(widget, **changes):
    """Rebuild the loaded set with the first curve's fields overridden."""
    original = widget.result_set()
    first = original.fits[0]
    edited = dataclasses.replace(first,
                                 result=dataclasses.replace(first.result,
                                                            **changes))
    widget._set = dataclasses.replace(original,
                                      fits=(edited,) + original.fits[1:])
    return widget._set


def test_a_bounded_ec50_with_an_open_interval_keeps_its_line_and_loses_the_band(
        fitted):
    """A shaded band is a claim about both ends of the interval.

    The profile walk is allowed to decline to close one side, and a result
    that carries an EC50 but only half an interval is a normal outcome, not a
    corrupt one. Drawing a band anyway would need a number the fit does not
    have; drawing nothing at all would hide an EC50 the fit *does* have. So
    the line stays and the band goes. The same figure is drawn first with
    both ends present, which is what puts the band on screen to begin with.
    """
    fitted._draw(0)
    axes = _axes(fitted)
    assert len(axes.patches) == 1, "a two-sided interval should shade a band"
    both_ends_text = [text.get_text() for text in axes.texts]

    _with_first_result(fitted, ec50_low=None)
    fitted._draw(0)
    axes = _axes(fitted)

    assert len(axes.patches) == 0
    assert len(_dashed(axes)) == 1, "the EC50 line itself must survive"
    assert [text.get_text() for text in axes.texts] == both_ends_text


def test_an_ec50_the_doses_never_bracket_is_an_arrow_at_the_edge(fitted):
    """An unbounded midpoint is drawn as a bound, not as a point estimate.

    ``EC50 > 27`` and ``EC50 = 27`` are different sentences and the picture
    has to make the same distinction the numbers do — otherwise a reader
    quotes the top dose as a potency. The marker moves to the edge of the
    tested range, in the warning colour, with the inequality in the label.
    """
    _with_first_result(fitted, ec50=None, ec50_bounded=False,
                       bound_direction="above")
    fitted._draw(0)
    axes = _axes(fitted)

    assert _dashed(axes) == [], "an open bound is not a dashed midpoint"
    dotted = [line for line in axes.lines if line.get_linestyle() == ":"]
    assert len(dotted) == 1
    edge = float(fitted.result_set().fits[0].result.dose_max)
    assert dotted[0].get_xdata()[0] == pytest.approx(edge)
    labels = [text.get_text() for text in axes.texts]
    assert len(labels) == 1
    assert labels[0].startswith("EC50 > ")


# ---------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------

class _CanvasWithoutATimer:
    """A canvas that never queued a draw, so it has nothing to cancel."""


def test_closing_the_screen_cancels_the_draw_it_queued(qtbot, two_gene_frame):
    """A draw that fires after teardown draws onto a destroyed canvas.

    The screen's canvas coalesces redraws onto a single-shot timer it owns.
    Closing the screen has to stop that timer, or the queued draw arrives
    with the figure already gone. Loading a frame queues exactly such a draw,
    which is what makes the cancellation observable rather than a no-op.
    """
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(two_gene_frame, label="synthetic")
    assert widget.canvas._draw_pending is True, (
        "set_frame should have queued a redraw to cancel")

    widget.close()

    assert widget.canvas._draw_pending is False
    assert widget.canvas._spacr_draw_timer.isActive() is False
    assert widget.is_busy() is False


def test_a_canvas_with_no_cancel_hook_does_not_break_the_close(
        qtbot, two_gene_frame):
    """Not every canvas the screen can be handed owns a draw timer.

    A plain matplotlib canvas, or a stand-in a host swapped in, has no
    ``cancel_pending_draw``; asking for it unconditionally would raise inside
    ``closeEvent`` and leave the worker pool un-shut-down, which is the exact
    condition Qt aborts the process over. The close has to complete and the
    jobs have to be stopped either way — the test above shows the same close
    calling the hook when the canvas does have one.
    """
    widget = DoseResponseScreen(threaded=False)
    qtbot.addWidget(widget)
    widget.set_frame(two_gene_frame, label="synthetic")
    widget.canvas = _CanvasWithoutATimer()

    assert widget.close() is True

    assert widget.isVisible() is False
    assert widget.active_jobs() == 0
    assert widget.is_busy() is False
    assert not hasattr(widget.canvas, "cancel_pending_draw")


# ---------------------------------------------------------------------------
# The other operand of each two-part guard
# ---------------------------------------------------------------------------

def test_the_name_guess_reads_every_option_before_it_settles(qtbot):
    """The dose guess is a scan, not a look at the head of the list.

    ``_refill`` is the one place a picker's contents are replaced, and its
    ``prefer`` hints are matched against *every* name in turn. A version that
    only inspected the first entry would leave the guess working on tables
    whose dose column happens to sort first and silently failing on every
    other table — the same screen guessing right or wrong depending on
    alphabetical luck. Both outcomes are driven here: a list where the hint
    is only found at the end, and a list where it is nowhere at all.
    """
    from PySide6.QtWidgets import QComboBox

    picker = QComboBox()
    qtbot.addWidget(picker)

    DoseResponseScreen._refill(
        picker, ["assay_level", "plate_row", "readout", "conc_uM"],
        prefer=("conc",))
    assert picker.currentText() == "conc_uM", (
        "the hint sits last, so the scan has to walk the three before it")

    DoseResponseScreen._refill(picker, ["alpha", "beta", "gamma"],
                               prefer=("conc",))
    assert picker.currentText() == "alpha", (
        "no name earns the hint, so the box keeps its own first entry")
    assert picker.count() == 3


def test_a_selection_the_last_fit_no_longer_has_draws_no_marker(fitted):
    """A stale row index must not put an EC50 line on the wrong curve.

    The table and the fit set are refilled independently, so an index left
    over from a larger plate can arrive at the figure after a smaller one has
    been fitted. Indexing the fits with it would raise, and clamping it would
    annotate a curve the user never selected — either way the picture would
    stop meaning what the grid says. An out-of-range selection draws the
    plate and no marker; the same figure with a real index is drawn first, so
    the missing line is a decision and not an empty axis.
    """
    fitted._draw(0)
    axes = _axes(fitted)
    assert len(_dashed(axes)) == 1, "an in-range row should mark its EC50"
    curves = len(axes.lines) - 1

    fitted._draw(len(fitted.result_set().fits) + 3)
    axes = _axes(fitted)

    assert _dashed(axes) == []
    assert [text.get_text() for text in axes.texts] == []
    assert len(axes.lines) == curves, "every curve is still drawn"
    # No index matched, so every curve is drawn in the unfocused weight --
    # the plate is still readable, it just has nothing singled out.
    assert {line.get_alpha() for line in axes.lines} == {0.25, 0.3}


def test_an_interval_open_at_the_top_also_loses_its_band(fitted):
    """Either end of the interval going missing has to drop the band.

    The profile walk can decline to close the upper side just as readily as
    the lower one — a compound whose top plateau is barely reached has an
    EC50 and an ``ec50_high`` of ``None``. Shading from the lower end to
    nowhere would draw a band whose right edge is invented, which is the one
    thing the screen must never do with a number the fit refused to give.
    The band is put on screen first with both ends present, so its absence
    afterwards is this result and not an empty figure.
    """
    fitted._draw(0)
    assert len(_axes(fitted).patches) == 1, "both ends present shades a band"

    _with_first_result(fitted, ec50_high=None)
    fitted._draw(0)
    axes = _axes(fitted)

    assert len(axes.patches) == 0
    assert len(_dashed(axes)) == 1, "the midpoint itself is still quotable"
    labels = [text.get_text() for text in axes.texts]
    assert len(labels) == 1 and labels[0].startswith("EC50 ")
    assert fitted.result_set().fits[0].result.ec50_low is not None, (
        "only the upper end was removed")
