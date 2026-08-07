"""Gates can be moved and resized after they are drawn.

The single biggest gap in the Gate Editor: a gate you cannot adjust is a
gate you redraw from scratch, and gating is how a screen becomes a
population.

Both operations return a NEW gate rather than mutating. These are frozen
dataclasses, a GateSet holds them by name, and an in-place edit would change
a gate something else is already holding a reference to.

The load-bearing property throughout is that the gate still means what it
looks like: after a move, the objects it selects are the objects under it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_spec import (
    GateError, PolygonGate, RectGate, ThresholdGate, gate_from_dict,
)


@pytest.fixture
def frame():
    xs, ys = np.meshgrid(np.arange(0, 21, 1.0), np.arange(0, 21, 1.0))
    return pd.DataFrame({"x_measure": xs.ravel(), "y_measure": ys.ravel()})


@pytest.fixture
def rect():
    return RectGate(name="box", x_column="x_measure", y_column="y_measure",
                    x_low=0, x_high=10, y_low=0, y_high=10)


@pytest.fixture
def poly():
    return PolygonGate(name="blob", x_column="x_measure",
                       y_column="y_measure",
                       vertices=((0, 0), (10, 0), (10, 10), (0, 10)))


# ---------------------------------------------------------------------------
# Moving
# ---------------------------------------------------------------------------

def test_moving_a_rectangle_moves_what_it_selects(rect, frame):
    """The point of the whole feature: the population follows the shape."""
    before = set(np.flatnonzero(rect.mask(frame)))
    moved = rect.translated(10.0, 10.0)
    after = set(np.flatnonzero(moved.mask(frame)))

    assert before != after
    assert len(after) == len(before), "a move must not change the size"
    # The moved gate covers the far corner; the original did not.
    assert moved.mask(frame)[(frame.x_measure == 20) & (frame.y_measure == 20)].all()
    assert not rect.mask(frame)[(frame.x_measure == 20) & (frame.y_measure == 20)].any()


def test_moving_a_polygon_moves_every_vertex(poly):
    moved = poly.translated(3.0, -2.0)
    assert moved.vertices == ((3.0, -2.0), (13.0, -2.0),
                              (13.0, 8.0), (3.0, 8.0))


def test_a_threshold_ignores_the_second_axis():
    """It is a cut on ONE column, so it has no y to move along."""
    gate = ThresholdGate(name="cut", column="x_measure", low=2, high=6)
    moved = gate.translated(3.0, 999.0)
    assert (moved.low, moved.high) == (5.0, 9.0)


def test_an_open_end_stays_open_when_moved():
    """None means "unbounded on this side". Adding to it would turn an open
    gate into a closed one the user never drew."""
    gate = ThresholdGate(name="cut", column="x_measure", low=5, high=None)
    moved = gate.translated(2.0, 0.0)
    assert moved.low == 7.0
    assert moved.high is None


def test_moving_returns_a_new_gate(rect):
    """Frozen dataclasses, held by name in a GateSet -- an in-place edit
    would change a gate something else already references."""
    moved = rect.translated(1.0, 1.0)
    assert moved is not rect
    assert rect.x_low == 0, "the original was mutated"


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------

def test_growing_a_rectangle_about_its_centre(rect):
    grown = rect.scaled(2.0)
    assert (grown.x_low, grown.x_high) == (-5.0, 15.0)
    assert (grown.y_low, grown.y_high) == (-5.0, 15.0)
    # The centre is what stays put.
    assert grown.centre() == rect.centre()


def test_shrinking_selects_a_subset(rect, frame):
    small = rect.scaled(0.5)
    assert set(np.flatnonzero(small.mask(frame))) < set(
        np.flatnonzero(rect.mask(frame)))


def test_resizing_about_a_grabbed_point_holds_that_point(poly):
    """"Click and pull" anchors on the opposite side, not the centre."""
    grown = poly.scaled(2.0, about=(0.0, 0.0))
    assert (0.0, 0.0) in grown.vertices
    assert (20.0, 20.0) in grown.vertices


def test_a_polygon_centre_is_the_vertex_centroid(poly):
    """Not the area centroid: for a strongly concave polygon that can sit
    OUTSIDE the shape, which makes a resize look like a move."""
    assert poly.centre() == (5.0, 5.0)


def test_a_half_open_threshold_has_no_centre_and_does_not_move():
    """A made-up centre would send the first resize somewhere arbitrary."""
    gate = ThresholdGate(name="cut", column="x_measure", low=5, high=None)
    assert gate.centre() == (None, None)
    assert gate.scaled(2.0) == gate


@pytest.mark.parametrize("factor", [0, -1, -0.5])
def test_a_non_positive_resize_is_refused(rect, factor):
    """Zero collapses the gate to a point and a negative turns it inside
    out. The arithmetic would accept both silently."""
    with pytest.raises(GateError, match="must be positive"):
        rect.scaled(factor)


# ---------------------------------------------------------------------------
# Per-vertex editing
# ---------------------------------------------------------------------------

def test_one_vertex_can_be_dragged(poly):
    edited = poly.with_vertex(2, 20.0, 20.0)
    assert edited.vertices[2] == (20.0, 20.0)
    assert edited.vertices[0] == poly.vertices[0], "other vertices moved"


def test_a_vertex_index_outside_the_polygon_is_refused(poly):
    """It would otherwise silently move a different corner than the one
    grabbed."""
    with pytest.raises(GateError, match="no vertex"):
        poly.with_vertex(9, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Everything still round-trips
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("edit", [
    lambda g: g.translated(3.0, 4.0),
    lambda g: g.scaled(1.5),
])
def test_an_edited_gate_still_serialises(rect, poly, frame, edit):
    """Persistence is free only if the edited gate is still an ordinary
    gate. If it were not, saved gates would come back in their pre-edit
    position."""
    for gate in (rect, poly):
        edited = edit(gate)
        restored = gate_from_dict(edited.to_dict())
        assert np.array_equal(restored.mask(frame), edited.mask(frame))


def test_editing_preserves_name_and_parent(poly):
    """The hierarchy must survive a drag, or moving a child re-parents it."""
    child = PolygonGate(name="child", parent="live cells",
                        x_column="x_measure", y_column="y_measure",
                        vertices=poly.vertices)
    for edited in (child.translated(1, 1), child.scaled(2.0)):
        assert edited.name == "child"
        assert edited.parent == "live cells"


# ---------------------------------------------------------------------------
# The canvas half: dragging a gate on screen
# ---------------------------------------------------------------------------

class _Event:
    """The parts of a matplotlib mouse event the handlers read."""

    def __init__(self, x, y, inaxes=True):
        self.xdata = x
        self.ydata = y
        self.inaxes = object() if inaxes else None


@pytest.fixture
def canvas(qtbot, frame, rect):
    pytest.importorskip("PySide6")
    from spacr.qt.widgets.gate_editor import GateCanvas
    from spacr.qt.widgets.gate_spec import GateSet

    widget = GateCanvas()
    qtbot.addWidget(widget)
    gates = GateSet()
    gates.add(rect)
    widget.set_gates(gates)
    # The canvas hit-tests against the loaded table, which it holds as
    # `_frame` -- `population()` derives the on-screen rows from it.
    widget._frame = frame
    return widget


def test_a_press_inside_a_gate_starts_a_move(canvas):
    canvas.set_tool("")
    canvas._on_press(_Event(5.0, 5.0))
    assert canvas._move_name == "box"


def test_a_press_outside_every_gate_starts_nothing(canvas):
    canvas.set_tool("")
    canvas._on_press(_Event(50.0, 50.0))
    assert canvas._move_name is None


def test_a_press_inside_a_gate_moves_it_whatever_tool_is_armed(canvas):
    """CHANGED 2026-08-07, and the change is the point.

    This used to assert that an armed tool suppressed the move. But the
    default tool is now RECTANGLE -- a drag draws a box without arming
    anything -- so under the old rule a gate could never be dragged unless
    the user first disarmed the tool they had drawn it with. Nobody thinks
    of that, and the gate looks stuck.
    """
    from spacr.qt.widgets.gate_editor import RECTANGLE

    canvas.set_tool(RECTANGLE)
    canvas._on_press(_Event(5.0, 5.0))
    assert canvas._move_name == "box"


def test_placing_a_polygon_vertex_does_not_drag_an_older_gate(canvas):
    """The one exception. Mid-polygon the user is placing vertices, and one
    that happens to land inside an existing gate must not drag it."""
    from spacr.qt.widgets.gate_editor import POLYGON

    canvas.set_tool(POLYGON)
    # The FIRST press with nothing pending is still allowed to grab a gate --
    # that is the "drag what is there" gesture. It is once vertices exist
    # that the user is committed to drawing.
    canvas._pending = [(2.0, 2.0)]
    canvas._on_press(_Event(6.0, 2.0))
    assert canvas._move_name is None
    assert len(canvas.pending_vertices()) >= 2


def test_the_drag_emits_the_moved_gate(canvas, qtbot):
    canvas.set_tool("")
    seen = []
    canvas.gate_edited.connect(seen.append)

    canvas._on_press(_Event(5.0, 5.0))
    canvas._on_release(_Event(9.0, 7.0))

    assert len(seen) == 1
    moved = seen[0]
    assert moved.name == "box"
    assert (moved.x_low, moved.x_high) == (4.0, 14.0)
    assert (moved.y_low, moved.y_high) == (2.0, 12.0)


def test_a_click_without_movement_selects_instead_of_moving(canvas):
    """A stray click must not mark the gate set dirty by moving it zero."""
    canvas.set_tool("")
    seen = []
    canvas.gate_edited.connect(seen.append)

    canvas._on_press(_Event(5.0, 5.0))
    canvas._on_release(_Event(5.0, 5.0))

    assert seen == []
    assert canvas.active_gate == "box"


def test_a_release_outside_the_axes_cancels(canvas):
    """Dragging off the plot must not teleport the gate to a nan."""
    canvas.set_tool("")
    seen = []
    canvas.gate_edited.connect(seen.append)

    canvas._on_press(_Event(5.0, 5.0))
    canvas._on_release(_Event(None, None, inaxes=False))

    assert seen == []
    assert canvas._move_name is None, "the drag state must always be cleared"


# ---------------------------------------------------------------------------
# The left panel
# ---------------------------------------------------------------------------

def test_filter_and_columns_are_one_section_not_two_tabs(qtbot):
    """They were separate tabs in a QTabWidget capped at 340px, and a panel
    needing more than that had nowhere to put it -- which is what read as
    elements overlapping. They are also the same job: both narrow what the
    scatter shows, so hiding one behind the other meant neither could be
    checked while using the other."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QLabel, QScrollArea, QTabWidget

    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)

    assert screen.findChildren(QTabWidget) == [], (
        "the side panel is still tabbed")
    headings = sorted(label.text() for label in screen.findChildren(QLabel)
                      if label.objectName() == "SectionHeading")
    assert headings == ["Columns", "Filter"]
    assert screen.findChildren(QScrollArea), (
        "the content is unbounded and the panel is not, so something has to "
        "scroll or something has to clip")


def test_the_side_panel_width_is_the_splitters_to_decide(qtbot):
    """A hard maximum is what made the cap unescapable: the user could not
    widen the column even when the content plainly needed it."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QScrollArea

    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    area = screen.findChildren(QScrollArea)[0]
    # Qt's "no maximum" sentinel. Anything smaller is a cap.
    assert area.maximumWidth() == 16777215
    assert area.minimumWidth() > 0, "it still needs a floor to be usable"


def test_the_side_panel_does_not_paint_the_window_colour(qtbot):
    """A QScrollArea's viewport auto-fills with `bg`, which is #000000 on
    dark -- a black slab beside the plot (INVARIANTS 2/3)."""
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QScrollArea

    from spacr.qt import theme
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    area = screen.findChildren(QScrollArea)[0]
    assert area.property(theme.TRANSPARENT_PROPERTY) is True


# ---------------------------------------------------------------------------
# Shapes beyond the rectangle
# ---------------------------------------------------------------------------

def test_an_ellipse_excludes_the_corners_a_rectangle_would_take(frame):
    """The reason to have one. A cloud of cells is round-ish, and a
    rectangle around it always takes corner debris with it."""
    from spacr.qt.widgets.gate_spec import EllipseGate, RectGate

    box = RectGate(name="box", x_column="x_measure", y_column="y_measure",
                   x_low=0, x_high=10, y_low=0, y_high=10)
    oval = EllipseGate.from_drag("oval", "x_measure", "y_measure",
                                 0, 0, 10, 10)

    inside_both = box.mask(frame) & oval.mask(frame)
    corner_only = box.mask(frame) & ~oval.mask(frame)
    assert inside_both.any()
    assert corner_only.any(), "the ellipse took everything the box did"
    # The (0, 0) corner is in the box and outside the oval.
    corner = (frame.x_measure == 0) & (frame.y_measure == 0)
    assert box.mask(frame)[corner].all()
    assert not oval.mask(frame)[corner].any()


def test_the_ellipse_is_inscribed_in_the_dragged_box():
    """It ends where the pointer did. A user who drags a box expects the
    shape to touch the corner they released at, not to extend past it."""
    from spacr.qt.widgets.gate_spec import EllipseGate

    oval = EllipseGate.from_drag("oval", "a", "b", 0, 0, 10, 6)
    assert oval.centre() == (5.0, 3.0)
    assert (oval.x_radius, oval.y_radius) == (5.0, 3.0)


def test_a_circle_is_just_an_ellipse_with_equal_radii():
    """No separate kind: a circle that cannot be squashed is a shape the
    user deletes the moment the axes are not comparable, and on two
    different measurements they never are."""
    from spacr.qt.widgets.gate_spec import EllipseGate

    circle = EllipseGate.from_drag("round", "a", "b", 0, 0, 8, 8)
    assert circle.x_radius == circle.y_radius == 4.0


def test_an_ellipse_moves_and_resizes_like_any_other_gate(frame):
    from spacr.qt.widgets.gate_spec import EllipseGate

    oval = EllipseGate.from_drag("oval", "x_measure", "y_measure",
                                 0, 0, 10, 10)
    moved = oval.translated(5.0, 5.0)
    assert moved.centre() == (10.0, 10.0)
    assert (moved.x_radius, moved.y_radius) == (oval.x_radius, oval.y_radius)

    grown = oval.scaled(2.0)
    assert grown.centre() == oval.centre(), "growing must not move it"
    assert grown.x_radius == oval.x_radius * 2


def test_a_zero_radius_ellipse_is_refused():
    """It would select nothing while looking like a gate."""
    from spacr.qt.widgets.gate_spec import EllipseGate, GateError

    with pytest.raises(GateError, match="selects nothing"):
        EllipseGate(name="flat", x_column="a", y_column="b",
                    x_centre=0, y_centre=0, x_radius=0, y_radius=1)


def test_an_ellipse_round_trips(frame):
    from spacr.qt.widgets.gate_spec import EllipseGate

    oval = EllipseGate.from_drag("oval", "x_measure", "y_measure",
                                 2, 2, 12, 8)
    restored = gate_from_dict(oval.to_dict())
    assert np.array_equal(restored.mask(frame), oval.mask(frame))


def test_the_oval_is_offered_as_a_tool():
    from spacr.qt.widgets.gate_editor import TOOL_LABELS
    from spacr.qt.widgets.gate_spec import ELLIPSE, GATE_KINDS

    assert ELLIPSE in GATE_KINDS
    assert ELLIPSE in TOOL_LABELS


# ---------------------------------------------------------------------------
# Gates belong to the axes they were drawn on
# ---------------------------------------------------------------------------

def test_a_gate_is_only_drawn_on_its_own_measurements(canvas, rect):
    """A gate is a statement about two named columns.

    Drawing it on a different pair puts the outline at coordinates that mean
    something else; not drawing it when the user returns to its own pair is
    how a gate seems to have vanished. Both directions are asserted.
    """
    from spacr.qt.widgets.graph_spec import GraphSpec

    canvas._spec = GraphSpec(x="x_measure", y="y_measure")
    assert canvas._gate_is_on_these_axes(rect) is True

    canvas._spec = GraphSpec(x="some_other", y="measurement")
    assert canvas._gate_is_on_these_axes(rect) is False

    # ...and back again: the gate reappears, it was never lost.
    canvas._spec = GraphSpec(x="x_measure", y="y_measure")
    assert canvas._gate_is_on_these_axes(rect) is True


def test_a_threshold_needs_only_its_own_column_on_screen(canvas):
    """A histogram puts it on x and a scatter may put it on either."""
    from spacr.qt.widgets.gate_spec import ThresholdGate
    from spacr.qt.widgets.graph_spec import GraphSpec

    cut = ThresholdGate(name="cut", column="x_measure", low=1, high=5)
    canvas._spec = GraphSpec(x="x_measure", y="y_measure")
    assert canvas._gate_is_on_these_axes(cut) is True
    canvas._spec = GraphSpec(x="y_measure", y="x_measure")
    assert canvas._gate_is_on_these_axes(cut) is True
    canvas._spec = GraphSpec(x="unrelated", y="also_unrelated")
    assert canvas._gate_is_on_these_axes(cut) is False


def test_the_gate_canvas_does_not_rescale_when_a_filter_is_applied():
    """The reported bug: applying a gate looked like zooming into it.

    Gating is the one place a filter must not move the axes -- rescaling to
    the rows a gate kept moves the view out from under the gate outline, so
    the gate appears to jump and becomes impossible to drag.
    """
    from spacr.qt.widgets.gate_editor import GateCanvas
    from spacr.qt.widgets.graph_builder import GraphCanvas

    assert GateCanvas.RESCALE_ON_FILTER is False
    # ...and an ordinary chart still follows its filter, which is what the
    # Graph Builder's own test asserts.
    assert GraphCanvas.RESCALE_ON_FILTER is True


def test_the_gate_list_has_its_own_handle(qtbot):
    """"the gate box should be independent."

    The gate list sits between the scatter and the filter column. It was in
    a box layout with a hard 320px cap, so it could not be resized at all:
    dragging the outer splitter moved the filter column and took the canvas
    AND the gate list with it as one block.
    """
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QSplitter

    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)

    pairs = {tuple(type(sp.widget(i)).__name__ for i in range(sp.count()))
             for sp in screen.findChildren(QSplitter)}
    assert ("GateCanvas", "GateTree") in pairs, pairs
    assert ("GateEditorPanel", "QScrollArea") in pairs, pairs


def test_the_gate_list_width_is_not_capped(qtbot):
    """A cap cannot be dragged past, so a gate whose name or statistics were
    wider than it had nowhere to be read."""
    pytest.importorskip("PySide6")
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    assert screen.gates.tree.maximumWidth() == 16777215
    assert screen.gates.tree.minimumWidth() > 0, (
        "without a floor the handle can hide the list entirely")


# ---------------------------------------------------------------------------
# The panel's own handlers
#
# These crashed in the real app while every test passed, because the tests
# exercised the CANVAS and the panel handlers were never called. `gates` is
# a property on both classes and both handlers called it as a method:
#
#     TypeError: 'GateSet' object is not callable
#
# on every single drag. So the handlers are called directly here.
# ---------------------------------------------------------------------------

@pytest.fixture
def panel(qtbot, frame, rect):
    pytest.importorskip("PySide6")
    from spacr.qt.linked_selection import LinkedSelection
    from spacr.qt.widgets.gate_editor import GateEditorPanel
    from spacr.qt.widgets.gate_spec import GateSet

    widget = GateEditorPanel(link=LinkedSelection())
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    gates = GateSet()
    gates.add(rect)
    widget.canvas.set_gates(gates)
    return widget


def test_the_panel_can_take_an_edited_gate_without_crashing(panel, rect):
    """The exact crash: a drag emits gate_edited, the panel handles it."""
    moved = rect.translated(3.0, 3.0)
    panel._on_gate_edited(moved)

    stored = panel.gates.get("box")
    assert (stored.x_low, stored.x_high) == (3.0, 13.0)


def test_the_panel_gates_property_is_not_callable(panel):
    """The shape of the bug, pinned. `gates` is a property on the panel AND
    on the canvas; anything calling it as a method raises on every use."""
    from spacr.qt.widgets.gate_spec import GateSet

    assert isinstance(panel.gates, GateSet)
    assert isinstance(panel.canvas.gates, GateSet)
    with pytest.raises(TypeError):
        panel.gates()


def test_the_drag_preview_is_the_shape_the_gate_will_be(qtbot):
    """"the oval looks like a square when dragged but does in fact generate
    an oval gate. the drag highlight area needs to correspond to the gate."

    The preview is what the user steers by, so it has to be the same shape
    AND the same geometry as the gate the release will produce -- a preview
    that merely looks elliptical but sits somewhere else is the same bug in
    a different costume. Both are asserted against
    EllipseGate.from_drag itself.
    """
    pytest.importorskip("PySide6")
    from matplotlib.patches import Ellipse, Rectangle

    from spacr.qt.widgets.gate_editor import GateCanvas
    from spacr.qt.widgets.gate_spec import ELLIPSE, RECTANGLE, EllipseGate

    canvas = GateCanvas()
    qtbot.addWidget(canvas)

    canvas.set_tool(ELLIPSE)
    patch = canvas._make_drag_patch(0.0, 0.0)
    assert isinstance(patch, Ellipse)
    canvas._update_drag_patch(patch, 0.0, 0.0, 10.0, 6.0)

    gate = EllipseGate.from_drag("oval", "a", "b", 0.0, 0.0, 10.0, 6.0)
    assert patch.get_center() == gate.centre()
    assert patch.get_width() == gate.x_radius * 2
    assert patch.get_height() == gate.y_radius * 2

    # The rectangle tool is untouched.
    canvas.set_tool(RECTANGLE)
    assert isinstance(canvas._make_drag_patch(0.0, 0.0), Rectangle)


def test_an_ordinary_chart_still_previews_a_rectangle(qtbot):
    """The hook lives on GraphCanvas, so a plain chart must be unaffected."""
    pytest.importorskip("PySide6")
    from matplotlib.patches import Rectangle

    from spacr.qt.widgets.graph_builder import GraphCanvas

    canvas = GraphCanvas()
    qtbot.addWidget(canvas)
    assert isinstance(canvas._make_drag_patch(0.0, 0.0), Rectangle)
