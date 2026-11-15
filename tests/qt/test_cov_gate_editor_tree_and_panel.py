"""The gate list and the gating surface, from the user's side.

Coverage of the widget half of ``spacr.qt.widgets.gate_editor``:
:class:`GateTree` and :class:`GateEditorPanel`.
The canvas half has its own files.

The gate-list rows and ticks, the cluster modal and ``run_cluster`` are
covered by ``tests/qt/test_gate_editor_tree_and_cluster.py``.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from PySide6.QtCore import Qt                                    # noqa: E402

from spacr.qt.linked_selection import LinkedSelection            # noqa: E402
from spacr.qt.widgets.gate_editor import (                       # noqa: E402
    GateEditorPanel, GateTree,
)
from spacr.qt.widgets.gate_spec import (                         # noqa: E402
    POLYGON, RECTANGLE, GateSet, RectGate,
)
from spacr.qt.widgets.graph_spec import CONTINUOUS, GraphSpec    # noqa: E402


# ---------------------------------------------------------------------------
# tables and fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grid() -> pd.DataFrame:
    """Four objects on a diagonal, so every gate below selects a listable set."""
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0],
                         "b": [1.0, 2.0, 3.0, 4.0],
                         "c": [1.0, 2.0, 3.0, 4.0]})


def _pair(name="one", parent=None, x="a", y="b", low=0.0, high=2.5):
    return RectGate(name=name, parent=parent, x_column=x, y_column=y,
                    x_low=low, x_high=high, y_low=low, y_high=high)


@pytest.fixture
def tree(qtbot) -> GateTree:
    widget = GateTree()
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def make_panel(qtbot):
    """Build a panel on ``frame``, plotting ``x`` against ``y``."""
    def build(frame, x="a", y="b"):
        panel = GateEditorPanel(link=LinkedSelection())
        qtbot.addWidget(panel)
        panel.set_frame(frame)
        roles = {c: CONTINUOUS for c in frame.columns
                 if pd.api.types.is_numeric_dtype(frame[c])}
        panel.set_spec(GraphSpec(x=x, y=y, roles=roles))
        return panel
    return build


@pytest.fixture
def panel(make_panel, grid):
    return make_panel(grid)


def test_a_refused_gate_snapshot_can_be_retried_without_changing_gates(
        panel, monkeypatch):
    before = panel._gates_snapshot()
    real = panel._gates.to_dict

    def unavailable():
        raise ValueError("strategy is not serializable yet")

    monkeypatch.setattr(panel._gates, "to_dict", unavailable)
    assert panel._gates_snapshot() == {}
    monkeypatch.setattr(panel._gates, "to_dict", real)
    assert panel._gates_snapshot() == before


# ---------------------------------------------------------------------------
# GateTree: the list itself
# ---------------------------------------------------------------------------


def test_unticking_a_gate_in_the_tree_reports_it_as_hidden(tree, grid):
    gates = GateSet()
    gates.add(_pair(name="one"))
    gates.add(_pair(name="two", high=3.5))
    tree.set_gates(gates, grid)
    seen = []
    tree.enabled_changed.connect(lambda name, on: seen.append((name, on)))

    tree.tree.topLevelItem(0).setCheckState(0, Qt.Unchecked)

    assert seen == [("one", False)]
    assert tree.is_enabled("one") is False
    assert tree.is_enabled("two") is True
    assert tree.is_enabled("never_drawn") is True, "an unknown gate is on"


# ---------------------------------------------------------------------------
# GateTree: the per-gate thresholds
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The cluster settings modal
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GateEditorPanel: the tool picker
# ---------------------------------------------------------------------------

def test_choosing_a_tool_in_the_dropdown_changes_what_a_drag_draws(panel):
    """The picker is the only way to change the tool, so a picker that does
    not reach the canvas is a picker that does nothing."""
    panel._tool.setCurrentIndex(panel._tool.findData(POLYGON))
    assert panel.canvas.gate_from_drag(1.0, 1.0, 3.0, 3.0) is None, \
        "a drag still drew a shape after Polygon was chosen"

    panel._tool.setCurrentIndex(panel._tool.findData(RECTANGLE))
    drawn = panel.canvas.gate_from_drag(1.0, 1.0, 3.0, 3.0)
    assert isinstance(drawn, RectGate)
    assert (drawn.x_low, drawn.x_high) == (1.0, 3.0)


# ---------------------------------------------------------------------------
# GateEditorPanel: clustering
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GateEditorPanel: drawing, selecting and publishing
# ---------------------------------------------------------------------------

def test_declining_to_name_a_gate_throws_the_drawing_away(panel, monkeypatch):
    """A gate is not a gate until it is named.

    The refused shape has to come off the plot too -- left there it looks
    like a gate that exists and cannot be selected.
    """
    panel.set_namer(lambda: "")
    repainted = []
    monkeypatch.setattr(panel.canvas, "render_now",
                        lambda: repainted.append(1))

    panel.canvas.gate_drawn.emit(
        panel.canvas.gate_from_drag(1.0, 1.0, 3.0, 3.0))

    assert panel.gates.is_empty
    assert repainted, "the abandoned outline was left on the plot"


# ---------------------------------------------------------------------------
# GateEditorPanel: the 3D controls
# ---------------------------------------------------------------------------

def test_the_panel_reports_the_plane_a_shape_would_land_on(panel):
    """The plane is picked, not read off the camera, so the panel can always
    say which one is armed."""
    assert panel.anchor_axis() == "z"

    panel._plane_buttons["x"].click()

    assert panel.anchor_axis() == "x"
    assert panel.canvas.anchor_axis() == "x"


def test_with_no_plane_ticked_the_editor_still_names_one(panel):
    """The three plane buttons are an exclusive group, so one is always
    ticked while the group holds. The fallback is what keeps `anchor_axis`
    answerable for a caller that reaches it when it does not."""
    group = panel._plane_buttons["z"].group()
    group.setExclusive(False)
    for button in panel._plane_buttons.values():
        button.setChecked(False)

    assert panel.anchor_axis() == "z"


def test_with_no_drag_mode_ticked_the_editor_still_names_one(panel):
    """As above: a drag has to mean something, and spinning is the safe one
    because it cannot create a gate nobody asked for."""
    group = panel._drag_buttons["spin"].group()
    group.setExclusive(False)
    for button in panel._drag_buttons.values():
        button.setChecked(False)

    assert panel.drag_mode() == "spin"


def test_from_view_needs_a_third_measurement_and_says_which(panel):
    """On a flat plot there is no depth to read, and inventing one would make
    a box that means something the user never framed."""
    panel.set_namer(lambda: "framed")

    panel.gate_from_view()

    assert panel.gates.is_empty
    assert "choose a Z" in panel.status()


def test_reset_view_undoes_a_zoom_and_a_spin_together(panel):
    """From the user's side there is one problem -- "the graph is not where
    it was" -- and having to know whether they zoomed or rotated to get out
    of it is a distinction only the implementation cares about."""
    panel.canvas._zoom = ((0.0, 1.0), (0.0, 1.0))
    panel.canvas._volume_zoom = 4.0
    panel.canvas._view_angles = (10.0, 20.0)

    panel.reset_view()

    assert panel.canvas._zoom is None
    assert panel.canvas._volume_zoom == 1.0
    assert panel.canvas._view_angles is None
