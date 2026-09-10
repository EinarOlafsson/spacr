"""The PCA panel's controls before, during and after a decomposition.

A user reaches these by doing ordinary things in an order nobody drew: hitting
Run before loading a table, ticking features off with the search box filled,
clicking a scree bar that is already on X, closing the window mid-fit. None of
them may raise, and each has one right answer.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt

from spacr.qt.widgets.graph_spec import SCATTER, GraphSpec
from spacr.qt.widgets.pca_model import PCASpec, pca
from spacr.qt.widgets.pca_view import (FeaturePicker, PCAPanel,
                                       PCAScoresCanvas, ScreePlot,
                                       arrow_scale)

pytestmark = pytest.mark.qt

FEATURES = ("area", "perimeter", "intensity")


def measurement_frame(n: int = 120, seed: int = 3) -> pd.DataFrame:
    """Two groups of objects with three correlated continuous measurements."""
    rng = np.random.default_rng(seed)
    shift = np.where(np.arange(n) < n // 2, -1.0, 1.0)
    size = shift + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": ["r1"] * n,
        "columnID": ["c1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": np.arange(n),
        "gene": np.where(shift < 0, "control", "knockdown"),
        "area": 800.0 + 60.0 * size,
        "perimeter": 110.0 + 7.0 * size + rng.normal(0.0, 0.5, n),
        "intensity": 5000.0 + 300.0 * rng.normal(0.0, 1.0, n),
    })


def a_result():
    frame = measurement_frame()
    return frame, pca(frame, PCASpec(features=FEATURES))


# ---------------------------------------------------------------------------
# The feature list
# ---------------------------------------------------------------------------

@pytest.fixture
def picker(qtbot):
    widget = FeaturePicker()
    qtbot.addWidget(widget)
    widget.set_frame(measurement_frame())
    return widget


def test_all_none_and_invert_act_on_what_the_search_is_showing(picker):
    """The buttons apply to the filtered list, not to every column.

    A user who typed a filter to work on a subset would otherwise wipe out
    ticks they cannot see -- and with four hundred features they would not
    find out until the fit came back with the wrong ones.
    """
    assert set(picker.selected()) == set(FEATURES)

    picker._search.setText("area")
    picker.select_none()
    assert set(picker.selected()) == {"perimeter", "intensity"}

    picker.invert()
    assert set(picker.selected()) == set(FEATURES)

    picker.select_none()
    picker._search.setText("")
    picker.select_all()
    assert set(picker.selected()) == set(FEATURES)


def test_unticking_a_row_updates_the_selection_and_the_count(picker, qtbot):
    """A click on the check box is the same edit as the None button.

    The label under the list is the only feedback that a tick landed, so it
    has to move with the box rather than only when the list is rebuilt.
    """
    changes = []
    picker.changed.connect(lambda: changes.append(picker.selected()))
    item = next(picker._list.item(i) for i in range(picker._list.count())
                if item_name(picker._list.item(i)) == "perimeter")

    item.setCheckState(Qt.Unchecked)

    assert "perimeter" not in picker.selected()
    assert picker._count.text() == "2 of 3 features ticked"
    assert changes and "perimeter" not in changes[-1]

    item.setCheckState(Qt.Checked)
    assert "perimeter" in picker.selected()
    assert picker._count.text() == "3 of 3 features ticked"


def item_name(item):
    return item.data(Qt.UserRole)


# ---------------------------------------------------------------------------
# The arrow ruler
# ---------------------------------------------------------------------------

def test_a_decomposition_with_no_components_draws_no_arrows():
    """No components, no plane, no ruler -- and no division by zero."""
    _frame, result = a_result()
    empty = replace(result, scores=result.scores[:, :0])

    assert empty.n_components == 0
    assert arrow_scale(empty, 0, 1, (-3.0, 3.0), (-3.0, 3.0)) == 0.0


def test_features_that_point_nowhere_in_the_plane_draw_no_arrows():
    """Every correlation zero means every arrow has length zero.

    Scaling that up to the axis would be a division by zero; the answer is
    to draw nothing, because there is nothing to point at.
    """
    _frame, result = a_result()
    flat = replace(result,
                   correlations=np.zeros_like(result.correlations))

    assert arrow_scale(flat, 0, 1, (-3.0, 3.0), (-3.0, 3.0)) == 0.0


def test_a_canvas_with_no_ruler_draws_no_circle(qtbot):
    """The reference circle is the ruler; with no scale there is none."""
    frame, result = a_result()
    canvas = PCAScoresCanvas()
    qtbot.addWidget(canvas)
    flat = replace(result, correlations=np.zeros_like(result.correlations))

    canvas.set_spec(GraphSpec(x="PC1", y="PC2", kind=SCATTER))
    canvas.set_result(flat, flat.scores_frame(frame))
    canvas.render_now()

    assert canvas.arrow_scale == 0.0
    circles = [patch for ax in canvas.panel_axes().values()
               for patch in ax.patches if hasattr(patch, "get_radius")]
    assert circles == []


def test_a_canvas_with_nothing_drawn_yet_adds_no_arrows(qtbot):
    """A result without a frame has no panels to draw arrows onto."""
    _frame, result = a_result()
    canvas = PCAScoresCanvas()
    qtbot.addWidget(canvas)

    canvas.set_spec(GraphSpec(x="PC1", y="PC2", kind=SCATTER))
    canvas.set_result(result, None)

    assert canvas.result is result
    assert canvas.panel_axes() == {}
    assert canvas.arrow_scale == 0.0


def test_the_plane_is_none_until_a_result_backs_it(qtbot):
    """Two component names are not a plane until the components exist.

    The spec is remembered across tables. Reopening a saved PC1-vs-PC5 view on
    a table that only supports three components must remove the arrows, not
    index off the end of the loadings.
    """
    frame, result = a_result()
    canvas = PCAScoresCanvas()
    qtbot.addWidget(canvas)
    canvas.set_spec(GraphSpec(x="PC1", y="PC2", kind=SCATTER))

    assert canvas.plane() is None, "no result yet"

    canvas.set_result(result, result.scores_frame(frame))
    assert canvas.plane() == (0, 1)

    canvas.set_spec(GraphSpec(x="PC1", y="PC9", kind=SCATTER))
    assert canvas.plane() is None
    assert canvas.arrow_scale == 0.0


# ---------------------------------------------------------------------------
# The scree plot
# ---------------------------------------------------------------------------

def test_clicking_a_bar_picks_that_component(qtbot):
    """A click on the scree names the component under the pointer."""
    frame, result = a_result()
    plot = ScreePlot()
    qtbot.addWidget(plot)
    plot.set_result(result)
    picked = []
    plot.component_picked.connect(picked.append)

    plot._on_click(SimpleNamespace(xdata=1.2, ydata=0.5))
    assert picked == [1]

    # Off the bars entirely: outside the plot area, and past the last bar.
    plot._on_click(SimpleNamespace(xdata=None, ydata=None))
    plot._on_click(SimpleNamespace(xdata=float(result.n_components + 3),
                                   ydata=0.5))
    assert picked == [1]


def test_a_scree_click_before_any_fit_picks_nothing(qtbot):
    """With no decomposition there is no component to pick."""
    plot = ScreePlot()
    qtbot.addWidget(plot)
    picked = []
    plot.component_picked.connect(picked.append)

    plot._on_click(SimpleNamespace(xdata=0.0, ydata=0.0))

    assert picked == []


# ---------------------------------------------------------------------------
# The panel
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qtbot):
    widget = PCAPanel()
    qtbot.addWidget(widget)
    return widget


def test_asking_for_a_pca_with_no_table_says_to_load_one(panel):
    """The refusal is a sentence in the panel, and it says the way out."""
    failures = []
    panel.failed.connect(failures.append)

    assert panel.recompute() is None

    assert panel.result is None
    assert failures == ["Load a table first."]
    assert panel.report.text() == "Load a table first."


def test_flipping_a_view_control_before_a_table_is_harmless(panel):
    """The biplot switch works before there is anything to plot."""
    panel._biplot.setChecked(False)

    assert panel.result is None
    assert panel.report.text() == "" or "Load a table" in panel.report.text()


def test_changing_the_scaling_refits(panel):
    """An option change is a new decomposition, not a redraw.

    Scaling changes the matrix that is decomposed, so the components
    themselves move; redrawing the old ones under a new label would be a
    picture of a fit nobody ran.
    """
    panel.set_frame(measurement_frame())
    was = panel.result.scaling
    before = panel.report.text()
    fits = []
    panel.computed.connect(fits.append)

    panel._scaling.setCurrentIndex(1 - panel._scaling.currentIndex())

    assert len(fits) == 1
    assert panel.result is fits[0]
    assert panel.result.scaling == panel.spec().scaling != was
    assert panel.report.text() != before, (
        "the report names the scaling the picture was made with")


def test_rebuilding_the_controls_does_not_refit_once_per_widget(panel):
    """While the panel is refilling its own pickers, option changes are quiet.

    Filling the boxes fires the same signals a user does. Without the guard a
    single table load would start one fit per widget touched -- on a
    200 000-row table that is seconds of frozen window each.
    """
    panel.set_frame(measurement_frame())
    fits = []
    panel.computed.connect(fits.append)

    panel._building = True
    try:
        panel._scaling.setCurrentIndex(1 - panel._scaling.currentIndex())
        panel._components.setValue(panel._components.value() + 1)
    finally:
        panel._building = False

    assert fits == []


def test_clicking_the_component_already_on_x_changes_nothing(panel):
    """Sliding PC1 onto Y to make room for PC1 would not be a plot."""
    panel.set_frame(measurement_frame())
    before = (panel.canvas.spec.x, panel.canvas.spec.y)
    assert before == ("PC1", "PC2")

    panel.scree.component_picked.emit(0)

    assert (panel.canvas.spec.x, panel.canvas.spec.y) == before


def test_a_scree_click_before_a_fit_leaves_the_panel_alone(panel):
    """The scree is empty then, but the signal can still arrive."""
    panel.scree.component_picked.emit(2)

    assert panel.result is None


def test_closing_the_panel_abandons_the_fit_in_flight(panel, monkeypatch):
    """A closed panel must not leave a decomposition running behind it.

    The worker holds the table and delivers into widgets that are being torn
    down. Shutting the runner down on close is what makes closing the window
    the end of the work.
    """
    stopped = []
    monkeypatch.setattr(panel._jobs, "shutdown",
                        lambda *a, **k: stopped.append(True))

    panel.close()

    assert stopped == [True]
