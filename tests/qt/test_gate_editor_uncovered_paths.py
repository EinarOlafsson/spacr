"""The Gate Editor's guarded seams, driven rather than described.

Every path here is one the widget takes when something around it -- the
theme registry, the figure canvas, the installed matplotlib -- refuses a
call the editor would like to make. The property under test is always the
same: the refusal costs the user the decoration it was asking for and
nothing else, so the canvas still draws, the gates still land where they
were dragged and the tree still counts them.
"""
from __future__ import annotations

import importlib.util
import logging
import sys

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_editor import (
    GateCanvas, GateEditorPanel, GateTree, QSS_NAME,
)
from spacr.qt.widgets.gate_spec import (
    GateSet, RectGate, ThresholdGate, RECTANGLE,
)
from spacr.qt.widgets.graph_spec import GraphSpec


GATE_EDITOR_SOURCE = "spacr/qt/widgets/gate_editor.py"


class _Mouse:
    """A press/motion/release carrying both data and pixel coordinates."""

    def __init__(self, ax, x_data, y_data, *, step=0):
        self.inaxes = ax
        self.xdata, self.ydata = x_data, y_data
        self.x, self.y = ax.transData.transform((x_data, y_data))
        self.button = 1
        self.step = step


def _frame():
    """A small deterministic table with a third measurement for the volume."""
    rng = np.random.default_rng(7)
    return pd.DataFrame({"a": rng.normal(0.0, 1.0, 60),
                         "b": rng.normal(0.0, 1.0, 60),
                         "c": rng.normal(0.0, 1.0, 60)})


def _canvas(qtbot, *, frame=None, gates=None):
    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(_frame() if frame is None else frame)
    canvas.set_spec(GraphSpec(x="a", y="b"))
    if gates is not None:
        canvas.set_gates(gates)
    return canvas


# ---------------------------------------------------------------------------
# The theme registry
# ---------------------------------------------------------------------------

def _load_gate_editor_afresh(name):
    """Execute the gate editor's source again under ``name``.

    A second module object, so the one the rest of the suite imported keeps
    its classes and its registration. The file is the same file, which is
    what makes the import-time branch below a real one.
    """
    from spacr.qt.widgets import gate_editor as installed

    spec = importlib.util.spec_from_file_location(name, installed.__file__)
    module = importlib.util.module_from_spec(spec)
    # Visible under its own name while it executes: ``@dataclass`` resolves
    # annotations through ``sys.modules[cls.__module__]``, and a module that
    # is not there yet fails on the first frozen dataclass in the file.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_the_editor_still_imports_when_the_theme_refuses_its_stylesheet(
        monkeypatch, caplog):
    """A registry that will not take the gate tree's block costs decoration.

    The block is the gate list's text colour. Losing it leaves the rows in
    Qt's default colour, which is a cosmetic loss; taking the module down
    with it would mean the Gate Editor screen could not be opened at all.
    """
    from spacr.qt import theme

    refused = []

    def refuse(name, fn, **kwargs):
        refused.append(name)
        raise RuntimeError("the registry is closed")

    monkeypatch.setattr(theme, "register_widget_qss", refuse)
    before = dict(theme._WIDGET_QSS)

    with caplog.at_level(logging.DEBUG, logger="spacr.qt.gate_editor"):
        module = _load_gate_editor_afresh(
            "spacr.qt.widgets._gate_editor_without_qss")

    assert refused == [QSS_NAME], "the module did not offer its own block"
    # The module is whole: its public surface is there and its stylesheet
    # function still builds a block, it simply never got registered.
    assert module.DEFAULT_TOOL == RECTANGLE
    qss = module._gate_tree_qss({"fg": "#ff0000", "bg": "#000000",
                                 "accent": "#00ff00", "fg_muted": "#888888"})
    assert "#ff0000" in qss and "QTreeWidget#GateHierarchy" in qss
    assert dict(theme._WIDGET_QSS) == before, \
        "a refused registration must not change the registry"
    assert any("gate tree stylesheet" in record.message
               for record in caplog.records)


# ---------------------------------------------------------------------------
# The figure canvas
# ---------------------------------------------------------------------------

def test_a_canvas_that_cannot_report_the_wheel_still_draws_and_gates(
        qtbot, monkeypatch):
    """A backend without ``scroll_event`` leaves the editor usable.

    The wheel zoom is the only thing lost. Drawing a gate is a press and a
    drag, so a canvas whose scroll connection was refused must still turn a
    sweep into a rectangle on the measurements on screen.
    """
    from spacr.qt.widgets.graph_builder import _canvas_class

    canvas_class = _canvas_class()
    original = canvas_class.mpl_connect
    refused = []

    def connect(self, event, slot):
        if event == "scroll_event":
            refused.append(event)
            raise ValueError("this backend reports no scroll events")
        return original(self, event, slot)

    monkeypatch.setattr(canvas_class, "mpl_connect", connect)

    canvas = _canvas(qtbot)
    assert refused == ["scroll_event"], "the wheel was never asked for"

    canvas.set_tool(RECTANGLE)
    gate = canvas.gate_from_drag(-1.0, -1.0, 1.0, 1.0, name="middle")
    assert isinstance(gate, RectGate)
    assert (gate.x_column, gate.y_column) == ("a", "b")
    assert canvas.axes_at(0, 0) is not None, "the canvas drew nothing"


# ---------------------------------------------------------------------------
# The installed matplotlib
# ---------------------------------------------------------------------------

def test_a_matplotlib_without_rotation_control_still_draws_the_volume(
        qtbot, monkeypatch):
    """The 3D view survives an Axes3D that cannot hand over its rotation.

    spaCR replaces matplotlib's free drag-rotation with its own axis lock,
    and asks the axes to stand down first. An axes that has no such method
    keeps matplotlib's rotation -- an axis lock that does not bite -- while
    the volume itself, the aura and the gates are drawn exactly as usual.
    """
    from mpl_toolkits.mplot3d import Axes3D

    monkeypatch.delattr(Axes3D, "disable_mouse_rotation", raising=True)

    canvas = _canvas(qtbot)
    canvas.set_mode("3D", z_column="c")

    ax = canvas.axes_at(0, 0)
    assert ax is not None and hasattr(ax, "get_zlim3d")
    assert canvas._in_volume(), "the volume was not drawn"
    assert ax.get_zlabel() == "c"
    assert canvas.anchor_plane() == ("a", "b", "c")


# ---------------------------------------------------------------------------
# What the gate overlay is allowed to draw
# ---------------------------------------------------------------------------

def test_turning_the_highlight_off_leaves_the_outline_behind(qtbot):
    """``highlight_gated`` decides the ringed objects, never the shape.

    The two halves of a gate on the plot are separable: the outline says
    where it is, the rings say which objects it took. Unticking the ringing
    must not take the outline with it, or the gate would look deleted.
    """
    from matplotlib.collections import PathCollection
    from matplotlib.patches import Polygon as MplPolygon

    from spacr.qt.widgets.gate_settings import GateEditorSettings

    gates = GateSet().add(RectGate(name="g", x_column="a", y_column="b",
                                   x_low=-1.0, x_high=1.0,
                                   y_low=-1.0, y_high=1.0))
    canvas = _canvas(qtbot, gates=gates)

    canvas.apply_settings(GateEditorSettings().replaced(highlight_gated=True))
    with_rings = canvas._artists
    assert any(isinstance(a, PathCollection) for a in with_rings), \
        "the objects inside the gate were never ringed"
    assert any(isinstance(a, MplPolygon) for a in with_rings)

    canvas.apply_settings(GateEditorSettings().replaced(highlight_gated=False))
    without_rings = canvas._artists
    assert not any(isinstance(a, PathCollection) for a in without_rings), \
        "the highlight was drawn although it is switched off"
    assert any(isinstance(a, MplPolygon) for a in without_rings), \
        "switching the highlight off also took the gate's outline"


# ---------------------------------------------------------------------------
# Reading a point off the volume
# ---------------------------------------------------------------------------

def test_a_measurement_too_flat_to_intersect_is_still_read_off_the_face(
        qtbot):
    """A depth axis of vanishing extent falls back to the affine reading.

    The camera ray is intersected with the chosen face to find the point
    under the cursor. On a measurement whose whole range is ~1e-14 -- a
    p-value column, or an intensity in absolute units -- the ray's component
    along that axis is numerically nothing and the intersection is
    meaningless. The endpoint-based reading takes over, so the cursor still
    names a point on the plane instead of the gesture being refused.
    """
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"a": rng.normal(0.0, 1.0, 60),
                          "b": rng.normal(0.0, 1.0, 60),
                          "c": rng.uniform(1e-15, 9e-15, 60)})
    canvas = _canvas(qtbot, frame=frame)
    canvas.set_mode("3D", z_column="c")
    canvas.set_anchor_axis("z")
    ax = canvas.axes_at(0, 0)
    assert canvas.anchor_plane() == ("a", "b", "c")

    span = ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
    assert 0.0 < span < 1e-13, \
        "the fixture's depth axis is not flat enough to exercise this"

    class _At:
        def __init__(self, x, y):
            self.x, self.y = float(x), float(y)

    left = canvas.screen_to_volume(_At(320.0, 250.0))
    right = canvas.screen_to_volume(_At(350.0, 250.0))
    assert left is not None and right is not None, \
        "a flat depth axis must not stop the plane being read"
    assert left[0] == right[0] == "a" and left[2] == right[2] == "b"

    x_limits, y_limits = ax.get_xlim3d(), ax.get_ylim3d()
    for reading in (left, right):
        assert x_limits[0] <= reading[1] <= x_limits[1], \
            "the cursor was read off the plane the volume is showing"
        assert y_limits[0] <= reading[3] <= y_limits[1]
    # A reading, not a constant: moving right along the screen moves the
    # point along the plane's first measurement.
    assert right[1] > left[1]


# ---------------------------------------------------------------------------
# The spin-speed wrapper
# ---------------------------------------------------------------------------

def test_a_spin_scaled_before_the_drag_began_leaves_the_pointer_alone(
        qtbot, caplog):
    """The speed wrapper scales a movement, and there is none to scale yet.

    matplotlib delivers motion over the volume whether or not a button is
    down, and it records the press position on the axes only once a drag has
    started. A movement with no anchor -- and a pointer that has left the
    canvas, whose coordinates are ``None`` -- must pass through untouched
    rather than being multiplied against a number that is not there.
    """
    from spacr.qt.widgets.gate_settings import GateEditorSettings

    canvas = _canvas(qtbot)
    canvas.apply_settings(GateEditorSettings().replaced(spin_speed=2.5))
    canvas.set_mode("3D", z_column="c")
    ax = canvas.axes_at(0, 0)
    assert getattr(ax._on_move, "_spacr_wrapped", False), \
        "the spin speed was never wrapped, so there is nothing to drive"
    assert not hasattr(ax, "_sx"), "the fixture has already begun a drag"

    before = (float(ax.elev), float(ax.azim))

    class _Motion:
        inaxes = None
        xdata = ydata = None
        x = y = None
        button = None

    event = _Motion()
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.gate_editor"):
        ax._on_move(event)

    assert event.x is None and event.y is None, \
        "the wrapper invented coordinates for a pointer that has none"
    assert (float(ax.elev), float(ax.azim)) == before, \
        "a movement with no drag behind it turned the volume"
    assert not [r for r in caplog.records if "scale the spin" in r.message], \
        "the wrapper tried the arithmetic and fell into its own handler"


# ---------------------------------------------------------------------------
# A volume gesture the view cannot read
# ---------------------------------------------------------------------------

def test_a_drag_on_a_plane_that_does_not_exist_spins_instead_of_drawing(
        qtbot):
    """Two axes on one measurement leave no plane to draw on, so it spins.

    The pickers are filled from one column list with nothing excluded, so a
    user can put the same measurement on X and Y. There is then no plane and
    no third axis to extend a footprint along. The drag is not swallowed:
    it turns the volume, which is the gesture that still means something.
    """
    canvas = _canvas(qtbot)
    canvas.set_spec(GraphSpec(x="a", y="a"))
    canvas.set_mode("3D", z_column="c")
    canvas.set_drag_mode("draw")
    ax = canvas.axes_at(0, 0)
    assert canvas._in_volume()
    assert canvas.anchor_plane() is None, \
        "the fixture still has three distinct measurements"

    class _At:
        def __init__(self, ax, x, y):
            self.inaxes = ax
            self.x, self.y = float(x), float(y)
            self.xdata = self.ydata = 0.0
            self.button = 1

    before = float(ax.azim)
    canvas._on_press(_At(ax, 300.0, 200.0))
    assert canvas._volume_drag is None, "a footprint was started regardless"
    assert canvas._spin_from == (300.0, 200.0)

    canvas._on_motion(_At(ax, 380.0, 200.0))
    assert float(ax.azim) != before, "the drag turned into nothing at all"


# ---------------------------------------------------------------------------
# Releases that commit nothing
# ---------------------------------------------------------------------------

def test_an_oval_dragged_with_no_width_offers_no_gate(qtbot):
    """A sweep with a zero side is not a shape, and nothing is proposed.

    An oval is inscribed in the swept box, so a drag straight up has a zero
    radius and no area. Naming it would ask the user to name nothing.
    """
    from spacr.qt.widgets.gate_spec import ELLIPSE, EllipseGate

    canvas = _canvas(qtbot)
    canvas.set_tool(ELLIPSE)
    ax = canvas.axes_at(0, 0)
    offered = []
    canvas.gate_drawn.connect(offered.append)

    canvas._on_press(_Mouse(ax, 0.0, -1.0))
    canvas._on_release(_Mouse(ax, 0.0, 1.0))
    assert offered == [], "a drag with no width was offered as a gate"

    # The same gesture with a width does produce one, so the refusal above is
    # about the shape and not about the gesture never arriving.
    canvas._on_press(_Mouse(ax, -1.0, -1.0))
    canvas._on_release(_Mouse(ax, 1.0, 1.0))
    assert len(offered) == 1 and isinstance(offered[0], EllipseGate)


def test_a_polygon_can_be_closed_without_announcing_itself(qtbot):
    """``close_polygon(emit=False)`` hands the gate back and stays quiet.

    The 3D path closes the shape to hold it while a second gesture gives it
    depth, and a scripted strategy closes one to keep the gate. Neither is
    the moment to ask the host to name a gate, so nothing is announced --
    while the pending vertices are consumed exactly as a normal close.
    """
    from spacr.qt.widgets.gate_spec import POLYGON, PolygonGate

    canvas = _canvas(qtbot)
    canvas.set_tool(POLYGON)
    ax = canvas.axes_at(0, 0)
    offered = []
    canvas.gate_drawn.connect(offered.append)

    for x, y in ((-1.0, -1.0), (1.0, -1.0), (0.0, 1.0)):
        canvas._on_press(_Mouse(ax, x, y))
    assert len(canvas.pending_vertices()) == 3

    gate = canvas.close_polygon(name="quiet", emit=False)

    assert isinstance(gate, PolygonGate)
    assert gate.name == "quiet"
    assert gate.vertices == ((-1.0, -1.0), (1.0, -1.0), (0.0, 1.0))
    assert canvas.pending_vertices() == (), "the vertices were not consumed"
    assert offered == [], "a silent close still asked the host to name it"


# ---------------------------------------------------------------------------
# The panel's controls
# ---------------------------------------------------------------------------

def test_a_default_tool_the_picker_does_not_offer_leaves_it_on_the_brush(
        qtbot, monkeypatch):
    """The tool picker survives a default it has no entry for.

    Four gate kinds are deliberately absent from the picker -- the three
    volume shapes and the composite, none of which is a drag. A default
    naming one of those has nothing to select, and the picker stays on its
    first entry rather than being driven to an index that does not exist.
    """
    from spacr.qt.widgets import gate_editor
    from spacr.qt.widgets.gate_spec import BOX

    monkeypatch.setattr(gate_editor, "DEFAULT_TOOL", BOX)

    panel = GateEditorPanel()
    qtbot.addWidget(panel)

    assert panel._tool.findData(BOX) == -1, \
        "the picker offers the volume shapes after all"
    assert panel._tool.currentIndex() == 0
    assert panel._tool.currentData() == "", "the picker landed somewhere odd"


def test_selecting_a_combined_gate_does_not_move_the_axes(qtbot):
    """A composite names no measurements of its own, so it asks for none.

    Selecting a gate normally asks the screen to show the two measurements
    it was drawn on. A combined gate's columns belong to its operands, and
    asking for the first of an empty list would be asking for nothing at
    all -- so the request is simply not made.
    """
    from spacr.qt.widgets.gate_spec import CompositeGate

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    panel.set_frame(_frame())
    panel.set_spec(GraphSpec(x="a", y="b"))

    gates = GateSet()
    gates.add(RectGate(name="left", x_column="a", y_column="b",
                       x_low=-3.0, x_high=0.0, y_low=-3.0, y_high=3.0))
    gates.add(RectGate(name="right", x_column="a", y_column="b",
                       x_low=0.0, x_high=3.0, y_low=-3.0, y_high=3.0))
    gates.add(CompositeGate(name="either", operation="union",
                            operands=("left", "right")))
    panel.set_gates(gates)

    asked = []
    panel.axes_requested.connect(lambda x, y: asked.append((x, y)))

    panel.tree.select("left")
    assert asked == [("a", "b")], "selecting a drawn gate asked for nothing"

    asked.clear()
    panel.tree.select("either")
    assert panel.tree.active_gate() == "either"
    assert asked == [], "a combined gate asked the screen to change axes"

    # Again straight at the handler, so a request built from an empty column
    # list would raise here rather than being swallowed by the signal.
    panel._on_active_changed("either")
    assert asked == []
