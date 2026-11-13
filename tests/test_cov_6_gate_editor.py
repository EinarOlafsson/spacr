"""What the gate canvas does when it cannot draw what it was asked for.

Every path here is decoration failing, and decoration is never load-bearing:
a log axis over data that reaches zero, a colour column with no numbers in
it, a gate whose measurements are not the two currently plotted, a click that
cannot be projected. In each case the plot has to keep its points and its
gates rather than take a traceback out of a paint path.

The distinction the tests hold is between "not drawn" and "drawn wrong". An
outline in the wrong units, a highlight from the wrong columns, or a gate
grabbable where it is invisible are all worse than the absence, so each is
asserted as an absence rather than as a fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.gate_editor import GateCanvas  # noqa: E402
from spacr.qt.widgets.gate_spec import (  # noqa: E402
    RectGate,
)
from spacr.qt.widgets.graph_spec import GraphSpec  # noqa: E402


@pytest.fixture()
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(5.0, 1.0, 200),
        "b": rng.normal(5.0, 1.0, 200),
        "c": rng.normal(5.0, 1.0, 200),
        "label": ["x"] * 200,
    })


@pytest.fixture()
def canvas(qtbot, frame):
    widget = GateCanvas()
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    widget.set_spec(GraphSpec(x="a", y="b"))
    return widget


def _axes(canvas):
    return list(canvas.panel_axes().values())[0]


# ---------------------------------------------------------------------------
# The armed tool
# ---------------------------------------------------------------------------

def test_the_armed_tool_is_readable(canvas):
    """The toolbar reads it back to show which button is down."""
    from spacr.qt.widgets.gate_editor import DEFAULT_TOOL

    assert canvas.tool == DEFAULT_TOOL


# ---------------------------------------------------------------------------
# Axis scales
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Colouring
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Binned drawing and density
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Voxels
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Gates that are not on these axes
# ---------------------------------------------------------------------------

def _rect(name="r1", x="a", y="b"):
    return RectGate(name=name, x_column=x, y_column=y,
                    x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)


def test_a_highlight_cannot_be_drawn_from_columns_the_table_lacks(canvas):
    """The outline still draws, so the gate is visible; the marks are not.

    A gate re-opened against a different measurements table names columns
    that are not there. Failing here is silent and total on purpose: a
    traceback out of a paint path would take the whole plot with it.
    """
    from spacr.qt.theme import active_palette

    axes = _axes(canvas)
    before = len(axes.collections)
    canvas._highlight(axes, _rect(),
                      pd.DataFrame({"other": [1.0, 2.0]}), active_palette())
    assert len(axes.collections) == before


def test_a_highlight_over_an_empty_table_draws_nothing(canvas):
    """A filter that kept no rows leaves the gate outlined and unmarked."""
    from spacr.qt.theme import active_palette

    axes = _axes(canvas)
    before = len(axes.collections)
    canvas._highlight(axes, _rect(), pd.DataFrame({"a": [], "b": []}),
                      active_palette())
    assert len(axes.collections) == before


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hit-testing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Outlines drawn directly, so the "not on these axes" cases are observable
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Hit-testing a shape that names no measurement
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Closing a volume polygon
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The xD button
# ---------------------------------------------------------------------------

def test_showing_a_projection_as_off_survives_a_panel_without_the_button(
        qtbot):
    """A projection can be refused before the toolbar has been built.

    The panel must not keep claiming a projection that did not happen, and
    the way it stops claiming it cannot depend on a button existing.
    """
    from spacr.qt.widgets.gate_editor import GateEditorPanel

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    button = panel._xd_button
    was_checked = button.isChecked()
    del panel._xd_button
    try:
        assert panel.set_projection_active(True) is None
        assert getattr(panel, "_xd_button", None) is None
    finally:
        panel._xd_button = button
    assert panel._xd_button.isChecked() == was_checked, (
        "the button must not have been reached at all")


def test_showing_a_projection_as_on_does_not_re_emit(qtbot):
    """Echoing it back would ask for the projection that just failed."""
    from spacr.qt.widgets.gate_editor import GateEditorPanel

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    seen = []
    panel.projection_requested.connect(seen.append)
    panel.set_projection_active(True)
    assert panel._xd_button.isChecked() is True
    assert seen == []
