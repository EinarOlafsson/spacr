"""Instruction 116's last line — the photograph is finally shown.

A run opened beside the loaded one is photographed at the moment it stops
being live, and `run_photograph(folder)` has handed that still back since
2026-08-18 with nothing painting it. This is where it lands: under the Runs
table, beside the row it belongs to.

The property that matters more than showing it is NOT showing it. Most rows
have no still, and an empty frame under the table would read as a run that
failed to draw rather than as one nobody has opened beside.

`FastPlot.pinned_limits()` rides along: 116 named the `getattr(volcano,
"_pinned")` in `plot_state` as its one private coupling and asked for a public
method to retire it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtGui import QColor, QPixmap                        # noqa: E402

from spacr.qt.widgets.sweep_runs import SweepRunsPanel           # noqa: E402


def _still(width: int = 320) -> QPixmap:
    pixmap = QPixmap(width, int(width * 0.75))
    pixmap.fill(QColor("#2E77BC"))
    return pixmap


def _panel(qtbot, stills=None):
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    if stills is not None:
        panel.set_photo_provider(lambda folder: stills.get(folder))
    return panel


# -- showing it -------------------------------------------------------------

def test_a_run_with_a_still_paints_it_under_the_table(qtbot):
    panel = _panel(qtbot, {"/x/ols_1": _still()})
    assert panel._show_photograph({"folder": "/x/ols_1"}) is True
    assert panel.photograph_shown() is not None


def test_a_run_nobody_has_opened_beside_shows_no_frame_at_all(qtbot):
    """An empty frame reads as a run that failed to draw."""
    panel = _panel(qtbot, {"/x/ols_1": _still()})
    assert panel._show_photograph({"folder": "/x/glm_9"}) is False
    assert panel.photograph_shown() is None


def test_the_frame_goes_away_again_when_the_selection_moves(qtbot):
    panel = _panel(qtbot, {"/x/ols_1": _still()})
    panel._show_photograph({"folder": "/x/ols_1"})
    assert panel.photograph_shown() is not None

    panel._show_photograph({"folder": "/x/glm_9"})
    assert panel.photograph_shown() is None


def test_with_no_provider_at_all_nothing_is_painted_and_nothing_raises(qtbot):
    panel = _panel(qtbot)
    assert panel._show_photograph({"folder": "/x/ols_1"}) is False
    assert panel.photograph_shown() is None


def test_a_provider_that_raises_costs_the_still_and_not_the_panel(qtbot):
    def angry(_folder):
        raise RuntimeError("Internal C++ object already deleted")

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_photo_provider(angry)
    assert panel._show_photograph({"folder": "/x/ols_1"}) is False


def test_a_record_with_no_folder_is_not_asked_about(qtbot):
    asked = []
    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_photo_provider(lambda folder: asked.append(folder))
    panel._show_photograph({"label": "a run with no folder"})
    panel._show_photograph(None)
    assert asked == []


def test_the_still_is_scaled_to_the_panel_rather_than_shown_at_capture_size(qtbot):
    panel = _panel(qtbot, {"/x/ols_1": _still(2000)})
    panel.resize(420, 600)
    panel._show_photograph({"folder": "/x/ols_1"})
    assert panel.photograph_shown().width() <= 420


def test_the_still_says_it_is_a_still(qtbot):
    """A picture of a plot that ignores clicks has to say why it does."""
    panel = _panel(qtbot, {"/x/ols_1": _still()})
    assert "still" in panel._photo.toolTip()
    assert "open the run" in panel._photo.toolTip().lower()


# -- the private coupling 116 named ----------------------------------------

def test_a_plot_answers_what_was_pinned_without_anyone_reaching_inside(qtbot):
    pytest.importorskip("pyqtgraph")
    from spacr.qt.widgets.fast_plots import FastPlot

    plot = FastPlot()
    qtbot.addWidget(plot)
    plot.plot.plot(np.arange(10), np.arange(10, dtype=float))

    assert plot.pinned_limits() == {"x": None, "y": None}
    plot.set_axis_limits(x=(0.0, 5.0))
    assert plot.pinned_limits()["x"] == (0.0, 5.0)
    assert plot.pinned_limits()["y"] is None


def test_the_answer_is_a_copy_so_a_caller_cannot_move_the_plot_through_it(qtbot):
    pytest.importorskip("pyqtgraph")
    from spacr.qt.widgets.fast_plots import FastPlot

    plot = FastPlot()
    qtbot.addWidget(plot)
    stashed = plot.pinned_limits()
    stashed["x"] = (99.0, 100.0)
    assert plot.pinned_limits()["x"] is None


def test_the_regression_panel_reads_the_plot_through_its_public_answer(qtbot):
    pytest.importorskip("pyqtgraph")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    asked = []
    panel.volcano.pinned_limits = lambda: asked.append(1) or {"x": (1.0, 2.0),
                                                              "y": None}
    state = panel.plot_state()
    assert asked, "plot_state did not go through pinned_limits()"
    assert state["x_limits"] == (1.0, 2.0)
