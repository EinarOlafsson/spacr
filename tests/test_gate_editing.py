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
    # Hit-testing only considers gates on the CURRENT axes, so the canvas
    # has to know which pair it is showing.
    from spacr.qt.widgets.graph_spec import GraphSpec
    widget._spec = GraphSpec(x="x_measure", y="y_measure")
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

    # NOT "no tabs anywhere": Filter and Search ARE tabs, which is what
    # instruction 31 asks for. What must not happen is Filter and COLUMNS
    # being split from each other -- they are the same job, and hiding one
    # behind the other meant neither could be checked while using the other.
    # So the assertion is that they share a page.
    tabs = screen.findChildren(QTabWidget)
    assert [tabs[0].tabText(i) for i in range(tabs[0].count())] == \
        ["Filter", "Search"]
    filter_page = tabs[0].widget(0)
    assert filter_page.isAncestorOf(screen.filters)
    assert filter_page.isAncestorOf(screen.formulas), (
        "Columns was split away from Filter")
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
    from spacr.qt.screens.gate_editor import GateEditorScreen

    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    # The SPLITTER'S CHILD, which is the tab widget since Filter and Search
    # became tabs. The scroll areas are inside it now, and their width is the
    # tabs' business rather than the splitter's.
    area = screen.side_tabs
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
    # Every scroll area on this screen, because they all sit beside the plot
    # and any one of them painting `bg` is the black slab.
    areas = screen.findChildren(QScrollArea)
    assert areas
    for area in areas:
        assert area.viewport().autoFillBackground() is False, (
            f"{area.objectName() or area} fills its viewport with the "
            f"window colour")


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
    # The console joined this splitter, so the body is three panes now. The
    # assertion is about the plot and the side panel each having a handle, not
    # about how many panes there happen to be -- pinning the count would fail
    # every time a pane is added, which is not what this test is for.
    # The side panel is a QTabWidget since Filter and Search became tabs.
    # The assertion is about the plot and the side panel each having a
    # HANDLE -- what the pane happens to be made of is not this test's
    # business, and pinning the class failed the moment the panel gained
    # tabs without anything about the handles changing.
    body = next(p for p in pairs if p[0] == "GateEditorPanel")
    assert len(body) >= 2, pairs


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


def test_closing_a_polygon_emits_exactly_one_gate(canvas):
    """"i get prompted for the name, then it zooms, then i get prompted again
    and always have to generate 2 identical gates."

    `close_polygon` already emits `gate_drawn`; the click-the-first-vertex
    wrapper emitted it a second time. One drawn polygon then prompted twice
    and produced two identical gates -- the second prompt arriving after the
    first gate had already been added and published, which is the "then it
    zooms" in between.
    """
    from spacr.qt.widgets.gate_editor import POLYGON
    from spacr.qt.widgets.graph_spec import GraphSpec

    canvas._spec = GraphSpec(x="x_measure", y="y_measure")
    canvas.set_tool(POLYGON)

    seen = []
    canvas.gate_drawn.connect(seen.append)

    canvas._pending = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
    canvas.close_polygon_now()

    assert len(seen) == 1, f"{len(seen)} gates emitted for one polygon"


def test_the_close_button_and_the_first_vertex_agree(canvas):
    """Two routes to the same act. Either emitting a different number of
    gates than the other is the bug above wearing the other hat."""
    from spacr.qt.widgets.gate_editor import POLYGON
    from spacr.qt.widgets.graph_spec import GraphSpec

    canvas._spec = GraphSpec(x="x_measure", y="y_measure")
    canvas.set_tool(POLYGON)

    counts = []
    for close in (canvas.close_polygon, canvas.close_polygon_now):
        seen = []
        handle = canvas.gate_drawn.connect(seen.append)
        canvas._pending = [(0.0, 0.0), (5.0, 0.0), (5.0, 5.0)]
        close()
        canvas.gate_drawn.disconnect(handle)
        counts.append(len(seen))
    assert counts == [1, 1], counts


# --- gates are overlays, not a view filter -------------------------------
#
# "pick x and y axes / draw a gate on the graph / be prompted to name the
# gate but never zoom into the gated data / be able now to select this gate
# in the gate panel section and toggle it on and off."
#
# What made it zoom was `GateCanvas.population()`: it returned the ACTIVE
# gate's population and `render_now` plots whatever it returns. Deleting the
# gate was the only thing that cleared the active name, which is exactly the
# symptom reported -- "the only way to get back to the main figure is to
# delete a gate".

def _canvas_with_gate(qtbot, monkeypatch):
    from spacr.qt.widgets.gate_editor import GateCanvas
    from spacr.qt.widgets.gate_spec import GateSet, RectGate
    from spacr.qt.widgets.graph_builder import GraphSpec
    import pandas as pd

    frame = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0], "b": [0.0, 1.0, 2.0, 3.0]})
    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(frame)
    canvas.set_spec(GraphSpec(x="a", y="b"))
    gates = GateSet().add(RectGate(name="g", x_column="a", y_column="b",
                                   x_low=-0.5, x_high=1.5,
                                   y_low=-0.5, y_high=1.5))
    return canvas, frame, gates


def test_selecting_a_gate_does_not_shrink_the_plot(qtbot, monkeypatch):
    """The whole table stays on screen while a gate is active.

    The strongest assertion available: the rows the canvas plots are the
    rows of the table, gate or no gate. Checking axis limits instead would
    pass on an empty plot.
    """
    canvas, frame, gates = _canvas_with_gate(qtbot, monkeypatch)
    canvas.set_gates(gates, active="g")
    on_screen = canvas.population()
    assert len(on_screen) == len(frame), (
        "selecting a gate replotted only its population -- that is the zoom")


def test_a_gate_can_be_toggled_off_and_back_on(qtbot, monkeypatch):
    canvas, frame, gates = _canvas_with_gate(qtbot, monkeypatch)
    canvas.set_gates(gates, active=None)
    assert canvas.enabled_gates == ("g",), "a new gate starts shown"

    canvas.set_gate_enabled("g", False)
    assert canvas.enabled_gates == ()
    assert "g" in canvas.gates, "hiding a gate must not delete it"
    assert len(canvas.population()) == len(frame), (
        "hiding a gate must not remove its rows from the plot either")

    canvas.set_gate_enabled("g", True)
    assert canvas.enabled_gates == ("g",)


def test_the_tick_in_the_tree_reaches_the_canvas(qtbot, monkeypatch):
    """The toggle is wired, not just implemented.

    Checked through the panel because the wiring is what broke before: the
    canvas grew the ability and nothing called it.
    """
    from PySide6.QtCore import Qt
    from spacr.qt.widgets.gate_editor import GateEditorPanel
    from spacr.qt.widgets.gate_spec import GateSet, RectGate
    from spacr.qt.widgets.graph_builder import GraphSpec
    import pandas as pd

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    panel.set_frame(pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]}))
    panel.canvas.set_spec(GraphSpec(x="a", y="b"))
    panel.set_gates(GateSet().add(RectGate(name="g", x_column="a", y_column="b",
                                           x_low=-1, x_high=1,
                                           y_low=-1, y_high=1)))

    item = panel.tree.tree.topLevelItem(0)
    assert item is not None and item.checkState(0) == Qt.Checked
    item.setCheckState(0, Qt.Unchecked)
    assert panel.canvas.enabled_gates == (), "unticking did not reach the canvas"
    item.setCheckState(0, Qt.Checked)
    assert panel.canvas.enabled_gates == ("g",)


def test_rebuilding_the_tree_does_not_report_toggles(qtbot, monkeypatch):
    """Setting a check state fires itemChanged; a rebuild must stay quiet.

    Otherwise every refresh -- and there is one after every edit -- reports
    every gate as freshly toggled by the user.
    """
    from spacr.qt.widgets.gate_editor import GateTree
    from spacr.qt.widgets.gate_spec import GateSet, RectGate
    import pandas as pd

    tree = GateTree()
    qtbot.addWidget(tree)
    seen = []
    tree.enabled_changed.connect(lambda n, on: seen.append((n, on)))
    tree.set_gates(GateSet().add(RectGate(name="g", x_column="a", y_column="b",
                                          x_low=-1, x_high=1,
                                          y_low=-1, y_high=1)),
                   pd.DataFrame({"a": [0.0], "b": [0.0]}))
    tree.refresh()
    assert seen == [], f"a rebuild reported toggles: {seen}"


# --- picking a gate up, and pulling its anchors --------------------------
#
# "when the gate is being moved it should be 'picked up' and placed. so a
# place holder in the shape of the gate should follow the mouse movement then
# when the mouse is released the gate is moved ... also i should be able to
# modify the gate by pulling on the corner anchorpoints or sides."

class _MouseEvent:
    """A matplotlib mouse event, with the pixel coordinates handles need."""

    def __init__(self, ax, x_data, y_data):
        self.inaxes = ax
        self.xdata, self.ydata = x_data, y_data
        self.x, self.y = ax.transData.transform((x_data, y_data))
        self.button = 1


def _rect_canvas(qtbot):
    from spacr.qt.widgets.gate_editor import GateCanvas
    from spacr.qt.widgets.gate_spec import GateSet, RectGate
    from spacr.qt.widgets.graph_builder import GraphSpec
    import pandas as pd

    canvas = GateCanvas()
    qtbot.addWidget(canvas)
    canvas.set_frame(pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0]}))
    canvas.set_spec(GraphSpec(x="a", y="b"))
    gate = RectGate(name="g", x_column="a", y_column="b",
                    x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0)
    canvas.set_gates(GateSet().add(gate))
    axes = canvas.panel_axes()
    return canvas, gate, list(axes.values())[0]


def test_a_rectangle_offers_corners_and_sides(qtbot):
    from spacr.qt.widgets.gate_spec import RectGate

    gate = RectGate(name="g", x_column="a", y_column="b",
                    x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0)
    handles = gate.handles((0.0, 10.0, 0.0, 10.0))
    corners = {(h.x, h.y) for h in handles if h.corner}
    assert corners == {(2.0, 2.0), (8.0, 2.0), (2.0, 8.0), (8.0, 8.0)}
    sides = {(h.x, h.y) for h in handles if not h.corner}
    assert sides == {(2.0, 5.0), (8.0, 5.0), (5.0, 2.0), (5.0, 8.0)}


def test_pulling_a_corner_moves_two_bounds_and_a_side_moves_one(qtbot):
    from spacr.qt.widgets.gate_spec import RectGate

    gate = RectGate(name="g", x_column="a", y_column="b",
                    x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0)
    pulled = gate.with_handle("x_high,y_high", 9.0, 9.0)
    assert (pulled.x_high, pulled.y_high) == (9.0, 9.0)
    assert (pulled.x_low, pulled.y_low) == (2.0, 2.0), "a corner moved the far side"

    pulled = gate.with_handle("x_low", 1.0, 999.0)
    assert pulled.x_low == 1.0
    assert (pulled.y_low, pulled.y_high) == (2.0, 8.0), "a side changed y"


def test_pulling_a_side_through_the_far_one_keeps_it_a_rectangle(qtbot):
    from spacr.qt.widgets.gate_spec import RectGate

    gate = RectGate(name="g", x_column="a", y_column="b",
                    x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0)
    pulled = gate.with_handle("x_low", 9.0, 0.0)
    assert pulled.x_low < pulled.x_high, "the rectangle was left inside out"
    assert (pulled.x_low, pulled.x_high) == (8.0, 9.0)


def test_an_oval_refuses_to_be_pulled_to_nothing(qtbot):
    from spacr.qt.widgets.gate_spec import EllipseGate

    gate = EllipseGate(name="o", x_column="a", y_column="b",
                       x_centre=5.0, y_centre=5.0, x_radius=2.0, y_radius=2.0)
    assert gate.with_handle("x_radius", 5.0, 5.0) is gate, (
        "a zero radius is not an ellipse, and EllipseGate refuses one")
    assert gate.with_handle("x_radius", 8.0, 5.0).x_radius == 3.0


def test_pressing_an_anchor_starts_a_resize_not_a_move(qtbot):
    """The anchor has to win. Every anchor sits on or inside its own gate,
    so testing the shape first would make resizing unreachable."""
    canvas, gate, ax = _rect_canvas(qtbot)
    canvas._on_press(_MouseEvent(ax, 8.0, 8.0))
    assert canvas._resize == ("g", "x_high,y_high")
    assert canvas._move_name is None, "grabbing a corner picked up the whole gate"


def test_pressing_inside_the_gate_still_moves_it(qtbot):
    canvas, gate, ax = _rect_canvas(qtbot)
    canvas._on_press(_MouseEvent(ax, 5.0, 5.0))
    assert canvas._resize is None
    assert canvas._move_name == "g"


def test_a_placeholder_follows_the_mouse_and_the_gate_does_not(qtbot):
    edits = []
    canvas, gate, ax = _rect_canvas(qtbot)
    canvas.gate_edited.connect(edits.append)

    canvas._on_press(_MouseEvent(ax, 5.0, 5.0))
    canvas._on_motion(_MouseEvent(ax, 6.0, 7.0))

    assert canvas._ghost, "nothing followed the mouse"
    assert canvas.gates.get("g").x_low == 2.0, (
        "the gate moved mid-drag; it is placed on RELEASE")
    assert edits == [], "an edit was committed before the mouse came up"

    canvas._on_release(_MouseEvent(ax, 6.0, 7.0))
    assert not canvas._ghost, "the placeholder outlived the drag"
    assert len(edits) == 1
    assert edits[0].x_low == 3.0 and edits[0].y_low == 4.0


def test_the_placeholder_shows_the_resize_that_release_will_commit(qtbot):
    """The ghost and the commit come from one function, so the dashed shape
    cannot promise something the release does not do."""
    edits = []
    canvas, gate, ax = _rect_canvas(qtbot)
    canvas.gate_edited.connect(edits.append)

    canvas._on_press(_MouseEvent(ax, 8.0, 8.0))
    moved = _MouseEvent(ax, 9.5, 9.5)
    preview = canvas._dragged_to(moved)
    canvas._on_motion(moved)
    assert canvas._ghost

    canvas._on_release(_MouseEvent(ax, 9.5, 9.5))
    assert len(edits) == 1
    assert (edits[0].x_high, edits[0].y_high) == (preview.x_high, preview.y_high)
    assert (edits[0].x_high, edits[0].y_high) == (9.5, 9.5)


def test_a_hidden_gate_cannot_be_grabbed(qtbot):
    """An invisible anchor that catches the mouse is indistinguishable from
    the plot being broken."""
    canvas, gate, ax = _rect_canvas(qtbot)
    canvas.set_gate_enabled("g", False)
    assert canvas.handle_at(_MouseEvent(ax, 8.0, 8.0)) is None


def test_an_oval_gate_is_actually_drawn(qtbot):
    """EllipseGate had no branch in `_outline` at all, so an oval was
    previewed while dragged and then vanished once it became a gate."""
    from spacr.qt.widgets.gate_spec import EllipseGate

    canvas, _gate, ax = _rect_canvas(qtbot)
    oval = EllipseGate(name="o", x_column="a", y_column="b",
                       x_centre=5.0, y_centre=5.0, x_radius=2.0, y_radius=3.0)
    points = canvas._gate_points(ax, oval)
    assert points, "an oval gate is not drawn"
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    assert round(max(xs), 6) == 7.0 and round(min(xs), 6) == 3.0
    assert round(max(ys), 6) == 8.0 and round(min(ys), 6) == 2.0
