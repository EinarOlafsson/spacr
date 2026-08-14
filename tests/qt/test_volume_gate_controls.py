"""The 3D gating controls, rebuilt. Instruction 52, reopened.

    "when in 3D mode there are 3 planes that you see, depending on what is
     chosen (y,x,z) the gate is drawn on one of these then propegated through
     the graph, there should be a drop down where Box gate is for Box gate,
     oval gate, curcular gate, etc. there should other options in tool for
     spinning and gate drawing."

The first attempt passed 97 tests and did not do the job. What it got wrong is
what this file pins: the plane was INFERRED from the camera instead of picked,
the shapes were HIDDEN from the dropdown instead of listed in it, and spinning
and drawing competed for one mouse button instead of being separate tools.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets.gate_editor import VOLUME_SHAPES, GateEditorPanel
from spacr.qt.widgets.gate_spec import BoxGate, CylinderGate, PrismGate


@pytest.fixture
def panel(qtbot):
    widget = GateEditorPanel()
    qtbot.addWidget(widget)
    from dataclasses import replace
    widget.canvas._spec = replace(widget.canvas._spec, x="a", y="b")
    widget.canvas._z_column = "c"
    widget.canvas._mode = "3D"
    return widget


# ---------------------------------------------------------------------------
# Three planes you PICK
# ---------------------------------------------------------------------------

def test_all_three_planes_are_offered(panel):
    assert set(panel._plane_buttons) == {"x", "y", "z"}


def test_picking_a_plane_sets_the_anchor(panel):
    panel._on_plane_picked("x")
    assert panel.canvas.anchor_axis() == "x"
    assert panel.canvas.anchor_plane() == ("b", "c", "a")


def test_only_one_plane_is_armed_at_a_time(panel):
    panel._plane_buttons["x"].click()
    assert panel._plane_buttons["x"].isChecked()
    panel._plane_buttons["y"].click()
    assert not panel._plane_buttons["x"].isChecked()


def test_picking_a_plane_arms_drawing(panel):
    """Choosing where to draw IS choosing to draw. Making the user find a
    second control afterwards is what reads as the feature not working."""
    assert panel.drag_mode() == "spin"
    panel._on_plane_picked("y")
    assert panel.drag_mode() == "draw"


# ---------------------------------------------------------------------------
# A dropdown of shapes, where a user looks for one
# ---------------------------------------------------------------------------

def test_the_dropdown_lists_every_volume_shape(panel):
    offered = [panel._volume_shape.itemData(i)
               for i in range(panel._volume_shape.count())]
    assert offered == [key for key, _label in VOLUME_SHAPES]
    assert offered == ["box", "oval", "circle", "polygon"]


def test_the_labels_are_the_words_the_request_used(panel):
    labels = [panel._volume_shape.itemText(i)
              for i in range(panel._volume_shape.count())]
    assert labels[:3] == ["Box gate", "Oval gate", "Circle gate"]


def test_choosing_a_shape_reaches_the_canvas(panel):
    panel._volume_shape.setCurrentIndex(1)
    assert panel.canvas.volume_shape() == "oval"


# ---------------------------------------------------------------------------
# Spin and draw are separate tools
# ---------------------------------------------------------------------------

def test_both_drag_modes_are_offered(panel):
    assert set(panel._drag_buttons) == {"spin", "draw"}


def test_spin_is_the_default(panel):
    assert panel.drag_mode() == "spin"


def test_switching_mode_reaches_the_canvas(panel):
    panel._on_drag_mode("draw")
    assert panel.canvas.drag_mode() == "draw"


# ---------------------------------------------------------------------------
# What a drag makes
# ---------------------------------------------------------------------------

class _Event:
    pass


def _drag(canvas, shape, start, end):
    canvas.set_volume_shape(shape)
    canvas._volume_drag = start
    canvas.screen_to_volume = lambda _e: end
    return canvas._gate_from_volume_drag(_Event())


def test_box_makes_a_box(panel):
    gate = _drag(panel.canvas, "box", ("a", 1.0, "b", 2.0), ("a", 3.0, "b", 6.0))
    assert isinstance(gate, BoxGate)


def test_oval_makes_a_cylinder(panel):
    gate = _drag(panel.canvas, "oval", ("a", 1.0, "b", 2.0), ("a", 3.0, "b", 6.0))
    assert isinstance(gate, CylinderGate)
    assert (gate.u_radius, gate.v_radius) == (1.0, 2.0)


def test_circle_is_round_on_the_plane(panel):
    """Both radii the same drag length. On two measurements with different
    units that is not round on screen, and it is still what 'circle' has to
    mean -- the alternative changes meaning when the axes rescale."""
    gate = _drag(panel.canvas, "circle", ("a", 0.0, "b", 0.0), ("a", 2.0, "b", 8.0))
    assert gate.u_radius == gate.v_radius == 4.0


# ---------------------------------------------------------------------------
# The slab you drag out
# ---------------------------------------------------------------------------

def test_no_depth_dragged_means_full_depth(panel):
    """What an undragged shape means, and what the 2D gate on that plane
    already meant."""
    gate = _drag(panel.canvas, "oval", ("a", 1.0, "b", 2.0), ("a", 3.0, "b", 6.0))
    assert (gate.axis_low, gate.axis_high) == (None, None)


def test_a_dragged_depth_makes_a_finite_slab(panel):
    panel.canvas.set_pending_depth(2.0, 7.0)
    gate = _drag(panel.canvas, "oval", ("a", 1.0, "b", 2.0), ("a", 3.0, "b", 6.0))
    assert (gate.axis_low, gate.axis_high) == (2.0, 7.0)


def test_a_dragged_depth_bounds_a_box_too(panel):
    panel.canvas.set_pending_depth(2.0, 7.0)
    gate = _drag(panel.canvas, "box", ("a", 1.0, "b", 2.0), ("a", 3.0, "b", 6.0))
    assert (gate.z_low, gate.z_high) == (2.0, 7.0)


def test_a_backwards_depth_drag_is_ordered(panel):
    panel.canvas.set_pending_depth(9.0, 3.0)
    assert panel.canvas.pending_depth() == (3.0, 9.0)


def test_a_dragged_depth_reaches_a_prism(panel, monkeypatch):
    from PySide6.QtWidgets import QInputDialog
    monkeypatch.setattr(QInputDialog, 'getText',
                        staticmethod(lambda *a, **k: ('p', True)))
    canvas = panel.canvas
    canvas.set_pending_depth(1.0, 4.0)
    canvas._tool = "polygon"
    canvas._pending = [(0.0, 0.0), (2.0, 0.0), (1.0, 2.0)]
    canvas._pending_plane = ("a", "b")
    gate = canvas.close_polygon(name="p")
    assert isinstance(gate, PrismGate)
    assert (gate.axis_low, gate.axis_high) == (1.0, 4.0)
