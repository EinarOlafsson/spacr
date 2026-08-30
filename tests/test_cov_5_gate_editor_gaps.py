"""What the gate canvas draws nothing for, and how a volume gate gets depth.

Two halves. In 2D, everything a gate cannot be drawn as: the wrong columns
plotted, a highlight that will not evaluate, a shape with no outline. Each is
asserted as an ABSENCE — an outline in the wrong units is worse than none.

In 3D, a gate is two gestures: a footprint on the picked plane, then a drag
along its normal for depth. The second gesture has to survive a view it
cannot project, a click that means "full depth", and a depth the gate itself
refuses.
"""
from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from spacr.qt.widgets.gate_editor import GateCanvas                # noqa: E402
from spacr.qt.widgets.gate_spec import (BoxGate, EllipseGate,      # noqa: E402
                                         GateError, GateSet,
                                         PolygonGate, RectGate,
                                         ThresholdGate)
from spacr.qt.widgets.graph_spec import GraphSpec                  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(5.0, 1.0, 200),
        "b": rng.normal(5.0, 1.0, 200),
        "c": rng.normal(5.0, 1.0, 200),
        "flat": np.zeros(200),
        "text": ["x"] * 200,
    })


@pytest.fixture()
def canvas(qtbot, frame):
    widget = GateCanvas()
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    widget.set_spec(GraphSpec(x="a", y="b"))
    return widget


@pytest.fixture()
def volume(qtbot, frame):
    widget = GateCanvas()
    qtbot.addWidget(widget)
    widget.set_frame(frame)
    widget.set_spec(GraphSpec(x="a", y="b"))
    widget.set_mode("3D", z_column="c")
    widget.set_anchor_axis("z")
    return widget


def _rect(name="box", x="a", y="b"):
    return RectGate(name=name, x_column=x, y_column=y,
                    x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)


# ---------------------------------------------------------------------------
# The armed tool
# ---------------------------------------------------------------------------

def test_arming_a_tool_nobody_has_is_refused(canvas):
    """The three tools are the three closed shapes; nothing else draws a gate."""
    from spacr.qt.widgets.gate_spec import GATE_KINDS

    assert canvas.tool in GATE_KINDS

    with pytest.raises(GateError, match="unknown gate tool"):
        canvas.set_tool("lasso")
    assert canvas.tool in GATE_KINDS, "a refused tool must not disarm the canvas"

    canvas.set_tool("polygon")
    assert canvas.tool == "polygon"
    canvas.set_tool("")
    assert canvas.tool == ""


def test_a_saved_default_tool_arms_only_an_unarmed_canvas(canvas):
    """A saved preference must not disarm the tool the user just picked."""
    settings = types.SimpleNamespace(default_tool="polygon")

    canvas.set_tool("")
    canvas.apply_settings(settings)
    assert canvas.tool == "polygon"

    canvas.set_tool("rectangle")
    canvas.apply_settings(settings)
    assert canvas.tool == "rectangle", "the saved default overrode a live tool"


# ---------------------------------------------------------------------------
# Axis scales the data cannot take
# ---------------------------------------------------------------------------

def test_a_log_axis_over_data_that_reaches_zero_is_skipped(canvas):
    """A log axis over non-positive data draws nothing, which reads as broken."""
    canvas.set_spec(GraphSpec(x="flat", y="b"), immediate=True)
    canvas._x_scale = "log"
    ax = canvas._figure.axes[0]

    canvas.decorate_axes(ax)

    assert ax.get_xscale() == "linear"


def test_a_scale_matplotlib_refuses_leaves_the_axis_alone(canvas):
    """A named scale can vanish between versions; the plot must survive it."""
    canvas.set_spec(GraphSpec(x="a", y="b"), immediate=True)
    canvas._x_scale = "symlog"
    ax = canvas._figure.axes[0]

    attempted = []

    def refuse(scale):
        attempted.append(scale)
        raise ValueError("unknown scale")

    ax.set_xscale = refuse

    result = canvas.decorate_axes(ax)

    assert result is None
    assert attempted == ["symlog"]
    assert ax.get_xscale() == "linear", "the rejected scale was not installed"


def test_a_column_the_frame_does_not_carry_is_not_positive(canvas):
    assert canvas._column_is_positive("nowhere") is False
    assert canvas._column_is_positive("flat") is False
    assert canvas._column_is_positive("text") is False
    assert canvas._column_is_positive("a") is True


# ---------------------------------------------------------------------------
# Colouring the cloud
# ---------------------------------------------------------------------------

def test_a_colour_column_with_no_numbers_falls_back_to_density(canvas):
    """A text column has no ramp; colouring by nothing would draw one flat hue."""
    canvas._colour_by = "text"
    canvas.set_spec(GraphSpec(x="a", y="b"), immediate=True)
    rows = canvas.population()
    x = rows["a"].to_numpy(float)
    y = rows["b"].to_numpy(float)

    values = canvas._colour_values(x, y, rows)

    assert values is not None
    assert len(values) == len(rows)
    assert np.isfinite(values).all()


def test_a_flat_colour_choice_asks_for_no_ramp_at_all(canvas):
    canvas._colour_by = "flat"
    rows = canvas.population()

    assert canvas._colour_values(rows["a"].to_numpy(float),
                                 rows["b"].to_numpy(float), rows) is None


def test_a_density_over_nothing_finite_is_all_zero(canvas):
    nan = np.full(5, np.nan)

    out = canvas._density(nan, nan)

    assert out.shape == (5,)
    assert not out.any()


def test_hexbin_is_a_resolution_the_binned_path_honours(canvas):
    """"hexbin did nothing" was hexbin living only in the points path."""
    from matplotlib.figure import Figure

    canvas._resolution = "hexbin"
    canvas._bins = 40
    ax = Figure().add_subplot(111)
    rng = np.random.default_rng(1)

    artist = canvas._draw_binned(ax, rng.normal(size=500), rng.normal(size=500))

    assert artist is not None
    assert len(ax.collections) == 1


def test_binning_nothing_draws_nothing(canvas):
    from matplotlib.figure import Figure

    assert canvas._draw_binned(Figure().add_subplot(111),
                                np.array([]), np.array([])) is None


# ---------------------------------------------------------------------------
# Gates that do not belong to the plotted measurements
# ---------------------------------------------------------------------------

def test_a_threshold_on_a_column_that_is_not_plotted_draws_no_line(canvas):
    """A dashed line at 4.0 on the wrong axis is a claim about the wrong number."""
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    canvas._artists = []

    canvas._outline(ax, ThresholdGate(name="cut", column="c", low=4.0),
                      None)

    assert canvas._artists == []
    assert len(ax.lines) == 0


def test_a_rectangle_from_another_pair_of_measurements_draws_nothing(canvas):
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    canvas._artists = []

    canvas._outline(ax, _rect(x="b", y="c"), None)

    assert canvas._artists == []
    assert len(ax.patches) == 0


def test_a_gate_naming_no_column_is_on_no_axes(canvas):
    stray = types.SimpleNamespace(name="odd", columns=())

    assert canvas._gate_is_on_these_axes(stray) is False


def test_a_shape_with_no_outline_draws_no_patch(canvas):
    """A gate kind the outline path does not know must not be drawn wrong."""
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    unknown = types.SimpleNamespace(name="odd", x_column="a", y_column="b")

    assert canvas._gate_points(ax, unknown) == []

    canvas._artists = []
    canvas._outline(ax, unknown, None)
    assert canvas._artists == []


def test_an_oval_is_outlined_as_a_polygon_rather_than_vanishing(canvas):
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    oval = EllipseGate(name="oval", x_column="a", y_column="b",
                       x_centre=5.0, y_centre=5.0, x_radius=1.0, y_radius=0.5)

    points = canvas._gate_points(ax, oval)

    assert len(points) == 64
    assert min(p[0] for p in points) == pytest.approx(4.0, abs=0.05)


# ---------------------------------------------------------------------------
# The ringed objects
# ---------------------------------------------------------------------------

def test_a_highlight_over_columns_that_are_not_here_rings_nothing(canvas):
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)
    canvas._artists = []
    other = pd.DataFrame({"c": [1.0, 2.0]})

    canvas._highlight(ax, _rect(), other, {"bg": "#000000"})

    assert canvas._artists == []


def test_a_highlight_that_will_not_evaluate_rings_nothing(canvas,
                                                          monkeypatch):
    """A traceback out of a paint path would take the whole plot with it."""
    from matplotlib.figure import Figure

    gate = _rect()
    canvas.set_gates(GateSet((gate,)))
    ax = Figure().add_subplot(111)
    canvas._artists = []

    def refuse(_frame, _name):
        raise ValueError("this gate cannot be evaluated here")

    monkeypatch.setattr(canvas._gates, "mask", refuse)

    canvas._highlight(ax, gate, canvas.population(), {"bg": "#000000"})

    assert canvas._artists == []


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------

def test_a_gate_from_another_pair_has_no_handles_here(canvas):
    """A handle where the gate is invisible is a grab nobody can see."""
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111)

    assert canvas._handles_for(ax, _rect(x="b", y="c")) == ()


def test_a_gate_whose_handles_raise_offers_none(canvas, monkeypatch):
    from matplotlib.figure import Figure

    gate = _rect()
    ax = Figure().add_subplot(111)

    def refuse(_view):
        raise RuntimeError("cannot place handles on an unbounded side")

    monkeypatch.setattr(type(gate), "handles",
                        lambda _self, _view: refuse(_view))

    assert canvas._handles_for(ax, gate) == ()

    canvas._artists = []
    canvas._draw_handles(ax, gate, {"bg": "#000000"})
    assert canvas._artists == []


# ---------------------------------------------------------------------------
# The volume: not enough points to bin
# ---------------------------------------------------------------------------

def test_a_small_cloud_is_drawn_as_dots_rather_than_voxels(volume):
    """Below the threshold an individual object can be seen and clicked."""
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111, projection="3d")
    volume._settings = types.SimpleNamespace(voxel_bins=8)
    small = np.zeros(10)

    assert volume._draw_voxels(ax, small, small, small) is False


def test_a_volume_whose_bins_are_all_empty_is_drawn_as_dots(volume,
                                                             monkeypatch):
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111, projection="3d")
    volume._settings = types.SimpleNamespace(voxel_bins=8)
    n = volume.VOXEL_THRESHOLD + 10
    values = np.zeros(n)

    monkeypatch.setattr(np, "histogramdd",
                        lambda *_a, **_k: (np.zeros((8, 8, 8)),
                                           [np.arange(9.0)] * 3))

    assert volume._draw_voxels(ax, values, values, values) is False


def test_a_volume_that_cannot_be_binned_is_drawn_as_dots(volume, monkeypatch):
    from matplotlib.figure import Figure

    ax = Figure().add_subplot(111, projection="3d")
    volume._settings = types.SimpleNamespace(voxel_bins=8)
    n = volume.VOXEL_THRESHOLD + 10
    values = np.zeros(n)

    def refuse(*_a, **_k):
        raise MemoryError("that grid does not fit")

    monkeypatch.setattr(np, "histogramdd", refuse)

    assert volume._draw_voxels(ax, values, values, values) is False


# ---------------------------------------------------------------------------
# The second gesture: depth
# ---------------------------------------------------------------------------

def test_a_footprint_drawn_in_2d_is_finished_at_once(canvas):
    """There is no third measurement in 2D, so there is no depth to ask for."""
    drawn = []
    canvas.gate_drawn.connect(drawn.append)
    gate = _rect()

    canvas._begin_volume_depth(gate)

    assert drawn == [gate]
    assert canvas._pending_volume_gate is None


def test_a_footprint_in_a_volume_asks_for_its_depth(volume):
    said = []
    volume.depth_requested.connect(said.append)
    gate = _rect()

    volume._begin_volume_depth(gate)

    assert volume._pending_volume_gate is gate
    assert volume._pending_volume_axis == "c"
    assert said and "Drag once more along c" in said[0]


def test_a_click_rather_than_a_drag_means_full_depth(volume):
    """The unbounded gate stays available without being the only outcome."""
    volume.render_now()
    volume._pending_volume_axis = "c"
    event = types.SimpleNamespace(x=100.0, y=100.0)

    assert volume._depth_bounds_from_drag((100.0, 100.0), event) == (None, None)


def test_a_drag_along_the_normal_becomes_a_bounded_depth(volume):
    volume.render_now()
    volume._pending_volume_axis = "c"
    ax = volume.axes_at(0, 0)
    assert ax is not None and hasattr(ax, "get_zlim")

    # A long drag in screen space: the projection decides how much of the
    # z range it covers, but it must be a real interval inside the limits.
    low, high = ax.get_zlim3d()
    bounds = volume._depth_bounds_from_drag((0.0, 0.0),
                                            types.SimpleNamespace(x=0.0,
                                                                  y=40.0))
    if bounds != (None, None):
        assert bounds[0] == pytest.approx(float(low))
        assert float(low) < bounds[1] <= float(high)


def test_a_depth_gesture_on_an_axis_the_plot_does_not_show_is_refused(volume):
    volume.render_now()
    volume._pending_volume_axis = "not_a_column"

    assert volume._depth_bounds_from_drag(
        (0.0, 0.0), types.SimpleNamespace(x=50.0, y=50.0)) is None


def test_a_depth_gesture_with_no_3d_axes_is_refused(canvas):
    canvas._pending_volume_axis = "c"

    assert canvas._depth_bounds_from_drag(
        (0.0, 0.0), types.SimpleNamespace(x=50.0, y=50.0)) is None


def test_finishing_a_depth_nobody_started_produces_no_gate(volume):
    assert volume._finish_volume_depth((0.0, 1.0)) is None

    volume._pending_volume_gate = _rect()
    volume._pending_volume_axis = ""
    assert volume._finish_volume_depth((0.0, 1.0)) is None


def test_a_depth_the_gate_refuses_is_reported_and_not_committed(volume):
    """An inverted or empty depth must not become a gate that selects nothing."""
    said = []
    drawn = []
    volume.depth_requested.connect(said.append)
    volume.gate_drawn.connect(drawn.append)
    gate = _rect()
    volume._begin_volume_depth(gate)
    said.clear()

    class _Refusing(type(gate)):
        pass

    def refuse(_self, _axis, _low, _high):
        raise GateError("a gate needs a low below its high")

    volume._pending_volume_gate = types.SimpleNamespace(
        with_threshold=lambda *_a: (_ for _ in ()).throw(
            GateError("a gate needs a low below its high")))

    assert volume._finish_volume_depth((5.0, 1.0)) is None
    assert drawn == []
    assert said and "low below its high" in said[0]
    assert volume._pending_volume_axis == "c", "the footprint was thrown away"


def test_a_finished_depth_becomes_a_volume_gate(volume):
    drawn = []
    volume.gate_drawn.connect(drawn.append)
    footprint = BoxGate(name="cube", x_column="a", y_column="b", z_column="c",
                        x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)
    volume._begin_volume_depth(footprint)

    gate = volume._finish_volume_depth((1.0, 8.0))

    assert gate is not None
    assert (gate.z_low, gate.z_high) == (1.0, 8.0)
    assert drawn == [gate]
    assert volume._pending_volume_gate is None
    assert volume._pending_volume_axis == ""
    assert volume.pending_depth() == (None, None)


def test_a_pending_depth_given_backwards_is_put_the_right_way_round(volume):
    volume.set_pending_depth(9.0, 2.0)

    assert volume.pending_depth() == (2.0, 9.0)


# ---------------------------------------------------------------------------
# Grabbing a gate, and the shape that follows the mouse
# ---------------------------------------------------------------------------

def test_a_handle_that_cannot_be_projected_is_not_grabbable(canvas):
    """An anchor the view cannot place must not catch the mouse anyway."""
    canvas.set_gates(GateSet((_rect(),)))
    canvas.render_now()
    ax = canvas._figure.axes[0]

    class _Broken:
        def transform(self, _point):
            raise ValueError("this transform is not invertible here")

    real = ax.transData
    ax.transData = _Broken()
    try:
        assert canvas.handle_at(
            types.SimpleNamespace(inaxes=ax, x=10.0, y=10.0)) is None
    finally:
        ax.transData = real


def test_a_synthetic_event_with_no_pixels_grabs_nothing(canvas):
    canvas.set_gates(GateSet((_rect(),)))
    canvas.render_now()
    ax = canvas._figure.axes[0]

    assert canvas.handle_at(types.SimpleNamespace(inaxes=None, x=1, y=1)) is None
    assert canvas.handle_at(
        types.SimpleNamespace(inaxes=ax, x=None, y=None)) is None


def test_a_ghost_artist_that_is_already_gone_is_dropped_quietly(canvas):
    """The figure may have been redrawn under the ghost between two moves."""
    class _Gone:
        def remove(self):
            raise ValueError("artist is not in the figure")

    canvas._ghost = [_Gone()]

    canvas._clear_ghost()

    assert canvas._ghost == []


def test_showing_no_ghost_clears_the_one_that_was_there(canvas):
    canvas.render_now()
    canvas._show_ghost(_rect())
    assert canvas._ghost

    canvas._show_ghost(None)

    assert canvas._ghost == []


def test_a_threshold_is_previewed_as_its_lines(canvas):
    canvas.set_spec(GraphSpec(x="a", y="b"), immediate=True)

    canvas._show_ghost(ThresholdGate(name="cut", column="a", low=4.0,
                                     high=6.0))

    assert len(canvas._ghost) == 2


def test_a_shape_with_no_outline_previews_nothing(canvas):
    canvas.render_now()

    canvas._show_ghost(types.SimpleNamespace(name="odd", x_column="a",
                                             y_column="b"))

    assert canvas._ghost == []


def test_a_drag_that_never_reached_the_axes_moves_nothing(canvas):
    canvas.set_gates(GateSet((_rect(),)))

    assert canvas._dragged_to(
        types.SimpleNamespace(inaxes=None, xdata=None, ydata=None)) is None


def test_a_drag_with_no_gate_picked_up_moves_nothing(canvas):
    canvas.set_gates(GateSet((_rect(),)))
    canvas.render_now()
    ax = canvas._figure.axes[0]
    canvas._resize = None
    canvas._move_name = ""
    canvas._move_from = None

    assert canvas._dragged_to(
        types.SimpleNamespace(inaxes=ax, xdata=5.0, ydata=5.0)) is None


def test_a_resize_the_gate_refuses_leaves_it_alone(canvas, monkeypatch):
    gate = _rect()
    canvas.set_gates(GateSet((gate,)))
    canvas.render_now()
    ax = canvas._figure.axes[0]
    canvas._resize = ("box", "not_a_handle")

    assert canvas._dragged_to(
        types.SimpleNamespace(inaxes=ax, xdata=5.0, ydata=5.0)) is None


def test_a_move_of_a_gate_that_is_gone_moves_nothing(canvas):
    canvas.set_gates(GateSet((_rect(),)))
    canvas.render_now()
    ax = canvas._figure.axes[0]
    canvas._resize = None
    canvas._move_name = "deleted"
    canvas._move_from = (5.0, 5.0)

    assert canvas._dragged_to(
        types.SimpleNamespace(inaxes=ax, xdata=6.0, ydata=6.0)) is None


# ---------------------------------------------------------------------------
# Hit-testing a click against the drawn gates
# ---------------------------------------------------------------------------

def test_a_gate_on_other_measurements_cannot_be_grabbed_invisibly(canvas):
    canvas.set_gates(GateSet((_rect(x="b", y="c"),)))
    canvas.render_now()

    assert canvas.gate_at(5.0, 5.0) is None


def test_a_gate_that_cannot_be_hit_tested_does_not_stop_the_ones_that_can(
        canvas, monkeypatch):
    """One unevaluable gate must not make every other gate unpickable."""
    good = _rect(name="good")
    canvas.set_gates(GateSet((good,)))
    canvas.render_now()
    assert canvas.gate_at(5.0, 5.0) == "good"

    def refuse(_self, _frame):
        raise ValueError("this gate cannot be evaluated on that probe")

    monkeypatch.setattr(type(good), "mask", refuse)
    assert canvas.gate_at(5.0, 5.0) is None


# ---------------------------------------------------------------------------
# The volume's mouse gestures
# ---------------------------------------------------------------------------

def _event(**kwargs):
    base = dict(inaxes=None, x=0.0, y=0.0, xdata=None, ydata=None)
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def test_the_volume_gestures_do_nothing_in_the_flat_view(canvas):
    """A 2D canvas must leave every event to the ordinary 2D handlers."""
    assert canvas._volume_press(_event()) is False
    assert canvas._volume_motion(_event()) is False
    assert canvas._volume_release(_event()) is False
    assert canvas._volume_scroll(_event()) is False


def test_a_press_outside_the_volume_axes_is_consumed_but_starts_nothing(
        volume):
    volume.render_now()

    assert volume._volume_press(_event(inaxes=None)) is True
    assert volume._volume_drag is None


def test_the_polygon_tool_takes_clicks_rather_than_drags(volume):
    """Click-per-vertex is `_on_press`'s job, so the drag must pass through."""
    volume.render_now()
    volume.set_drag_mode("draw")
    volume.set_volume_shape("polygon")
    ax = volume.axes_at(0, 0)

    assert volume._volume_press(_event(inaxes=ax, x=10.0, y=10.0)) is False


def test_a_press_while_a_footprint_waits_starts_the_depth_drag(volume):
    volume.render_now()
    volume.set_drag_mode("draw")
    volume.set_volume_shape("box")
    volume._pending_volume_gate = BoxGate(
        name="cube", x_column="a", y_column="b", z_column="c",
        x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)
    ax = volume.axes_at(0, 0)

    assert volume._volume_press(_event(inaxes=ax, x=12.0, y=34.0)) is True
    assert volume._depth_drag_from == (12.0, 34.0)


def test_a_spin_drag_turns_the_volume_about_the_locked_axis(volume):
    volume.render_now()
    volume.set_drag_mode("spin")
    ax = volume.axes_at(0, 0)
    assert volume._volume_press(_event(inaxes=ax, x=0.0, y=0.0)) is True
    before = (float(ax.elev), float(ax.azim))

    volume.set_spin_axis("z")
    volume._volume_motion(_event(inaxes=ax, x=40.0, y=0.0))
    assert float(ax.elev) == pytest.approx(before[0])
    assert float(ax.azim) != pytest.approx(before[1])

    volume.set_spin_axis("x")
    volume._volume_motion(_event(inaxes=ax, x=40.0, y=30.0))
    assert float(ax.elev) != pytest.approx(before[0])

    volume.set_spin_axis("")
    turned = (float(ax.elev), float(ax.azim))
    volume._volume_motion(_event(inaxes=ax, x=80.0, y=60.0))
    assert (float(ax.elev), float(ax.azim)) != turned


def test_a_motion_with_no_gesture_in_flight_turns_nothing(volume):
    volume.render_now()
    ax = volume.axes_at(0, 0)
    volume._spin_from = None

    assert volume._volume_motion(_event(inaxes=ax)) is True
    assert volume._view_angles is None


def test_the_depth_drag_reports_what_it_would_make(volume):
    volume.render_now()
    volume._pending_volume_axis = "c"
    volume._depth_drag_from = (0.0, 0.0)
    said = []
    volume.depth_requested.connect(said.append)

    volume._volume_motion(_event(x=0.0, y=0.0))

    assert said and "Full depth selected" in said[-1]


def test_a_release_of_a_depth_the_view_cannot_read_says_how_to_fix_it(volume):
    volume.render_now()
    volume._pending_volume_axis = "not_a_column"
    volume._depth_drag_from = (0.0, 0.0)
    said = []
    volume.depth_requested.connect(said.append)

    assert volume._volume_release(_event(x=50.0, y=50.0)) is True

    assert volume._depth_drag_from is None
    assert said and "turn the normal axis into view" in said[-1]


def test_a_sweep_that_described_no_volume_leaves_the_plot_redrawn(volume,
                                                                  monkeypatch):
    """A drag with no extent is not a gate; the aura has to come off."""
    volume.render_now()
    volume._volume_drag = ("a", 5.0, "b", 5.0)
    monkeypatch.setattr(volume, "screen_to_volume", lambda _event: None)

    assert volume._volume_release(_event(inaxes=None, x=0.0, y=0.0)) is True

    assert volume._volume_drag is None
    assert volume._pending_volume_gate is None


def test_scrolling_a_volume_with_no_third_axis_changes_no_zoom(canvas):
    canvas.set_mode("3D", z_column="c")
    canvas._mode = "3D"
    canvas.set_spec(GraphSpec(x="a", y=""), immediate=True)
    before = canvas._volume_zoom

    canvas._volume_scroll(_event(step=1))

    assert canvas._volume_zoom == before


def test_scrolling_the_volume_zooms_it(volume):
    volume.render_now()
    before = volume._volume_zoom

    assert volume._volume_scroll(_event(step=1)) is True

    assert volume._volume_zoom > before


def test_a_preview_the_view_cannot_place_draws_no_rectangle(volume):
    volume.render_now()
    volume._volume_drag = None

    volume._show_volume_drag(_event(x=0.0, y=0.0))

    assert volume._ghost == []


def test_a_plane_naming_the_same_measurement_twice_previews_nothing(volume):
    """The pickers allow it, so every gesture has to refuse it rather than raise."""
    volume.set_spec(GraphSpec(x="a", y="a"), immediate=True)
    volume.set_mode("3D", z_column="a")
    volume.render_now()
    volume._volume_drag = ("a", 4.0, "a", 4.0)

    volume._show_volume_drag(_event(x=10.0, y=10.0))
    assert volume._ghost == []

    assert volume._gate_from_volume_drag(_event(x=10.0, y=10.0)) is None


def test_a_sweep_with_no_extent_is_not_a_gate(volume, monkeypatch):
    """A press and release at one point names no region."""
    volume.render_now()
    start = ("a", 5.0, "b", 5.0)
    volume._volume_drag = start
    monkeypatch.setattr(volume, "screen_to_volume", lambda _event: start)

    assert volume._gate_from_volume_drag(_event(inaxes=None)) is None

    volume._volume_drag = None
    assert volume._gate_from_volume_drag(_event(inaxes=None)) is None


def test_a_vertex_on_a_plane_that_is_no_longer_shown_has_no_place(volume):
    """Changing a picker mid-polygon leaves the trail with nowhere to land."""
    volume.render_now()
    ax = volume.axes_at(0, 0)
    volume._pending_plane = ("a", "b")
    volume.set_spec(GraphSpec(x="a", y="b"), immediate=True)
    volume.set_mode("3D", z_column="a")
    volume.render_now()

    assert volume._volume_face_point(volume.axes_at(0, 0), 5.0, 5.0) is None

    volume._pending_plane = ()
    assert volume._volume_face_point(ax, 5.0, 5.0) is None


# ---------------------------------------------------------------------------
# Reading a click back into the volume when the camera will not invert
# ---------------------------------------------------------------------------

def test_a_click_falls_back_to_the_affine_reading_when_the_ray_will_not_invert(
        volume, monkeypatch):
    """matplotlib has moved its projection matrix more than once.

    The exact answer inverts the camera ray; when that path is unavailable the
    endpoint-based affine inverse still lands the click on the picked face,
    which is better than a drag that silently refuses.
    """
    from mpl_toolkits.mplot3d import proj3d

    volume.render_now()

    def refuse(*_args, **_kwargs):
        raise AttributeError("this matplotlib spells its projection elsewhere")

    monkeypatch.setattr(proj3d, "inv_transform", refuse)

    read = volume.screen_to_volume(_event(x=120.0, y=140.0))

    assert read is not None
    first, x, second, y = read
    assert (first, second) == ("a", "b")
    assert np.isfinite(x) and np.isfinite(y)


# ---------------------------------------------------------------------------
# Depth the view cannot measure
# ---------------------------------------------------------------------------

def _flat_limits(volume, monkeypatch, low=5.0, high=5.0):
    ax = volume.axes_at(0, 0)
    monkeypatch.setattr(ax, "get_zlim3d", lambda: (low, high))
    return ax


def test_a_measurement_with_no_range_has_no_depth_to_drag(volume,
                                                           monkeypatch):
    """A z axis whose limits coincide cannot express any fraction of itself."""
    volume.render_now()
    volume._pending_volume_axis = "c"
    _flat_limits(volume, monkeypatch)

    assert volume._depth_bounds_from_drag(
        (0.0, 0.0), _event(x=80.0, y=80.0)) is None


def test_a_depth_gesture_the_projection_refuses_is_refused(volume,
                                                            monkeypatch):
    volume.render_now()
    volume._pending_volume_axis = "c"
    ax = volume.axes_at(0, 0)

    class _Broken:
        def transform(self, _point):
            raise ValueError("cannot project that point")

    real = ax.transData
    ax.transData = _Broken()
    try:
        assert volume._depth_bounds_from_drag(
            (0.0, 0.0), _event(x=80.0, y=80.0)) is None
    finally:
        ax.transData = real


def test_an_edge_on_normal_axis_has_no_depth_to_read(volume, monkeypatch):
    """Seen edge-on the axis collapses to a point; no drag along it means anything."""
    volume.render_now()
    volume._pending_volume_axis = "c"
    ax = volume.axes_at(0, 0)

    class _Collapsed:
        def transform(self, _point):
            return (100.0, 100.0)

    real = ax.transData
    ax.transData = _Collapsed()
    try:
        assert volume._depth_bounds_from_drag(
            (0.0, 0.0), _event(x=180.0, y=180.0)) is None
    finally:
        ax.transData = real


def test_a_drag_that_spans_the_whole_axis_means_full_depth(volume):
    """Dragging past the end is the same statement as saying nothing about z."""
    volume.render_now()
    volume._pending_volume_axis = "c"
    ax = volume.axes_at(0, 0)

    class _Wide:
        def __init__(self, real):
            self._real = real
            self._n = 0

        def transform(self, point):
            self._n += 1
            return (0.0, 0.0) if self._n == 1 else (0.0, 10.0)

    real = ax.transData
    ax.transData = _Wide(real)
    try:
        # A 100-pixel drag along a 10-pixel axis is well past the far end.
        assert volume._depth_bounds_from_drag(
            (0.0, 0.0), _event(x=0.0, y=100.0)) == (None, None)
    finally:
        ax.transData = real


def test_a_partial_drag_becomes_a_slab_and_is_announced(volume):
    """The panel has to say what the release will make, in the data's own units."""
    volume.render_now()
    volume._pending_volume_axis = "c"
    ax = volume.axes_at(0, 0)
    low, high = (float(v) for v in ax.get_zlim3d())

    class _Wide:
        def __init__(self):
            self._n = 0

        def transform(self, _point):
            self._n += 1
            return (0.0, 0.0) if self._n == 1 else (0.0, 100.0)

    real = ax.transData
    ax.transData = _Wide()
    try:
        bounds = volume._depth_bounds_from_drag((0.0, 0.0),
                                                _event(x=0.0, y=50.0))
        assert bounds is not None
        assert bounds[0] == pytest.approx(low)
        assert low < bounds[1] < high

        said = []
        volume.depth_requested.connect(said.append)
        volume._depth_drag_from = (0.0, 0.0)
        ax.transData = _Wide()
        volume._volume_motion(_event(x=0.0, y=50.0))
    finally:
        ax.transData = real

    assert said and "release to create the gate" in said[-1]
    assert "Depth" in said[-1]


def test_a_volume_whose_axes_are_not_a_3d_plot_turns_nothing(volume,
                                                              monkeypatch):
    """A stand-in that reports depth but cannot be turned must not be turned."""
    class _NotAnAxes3D:
        def get_zlim(self):
            return (0.0, 1.0)

    volume.render_now()
    volume._spin_from = (0.0, 0.0)
    monkeypatch.setattr(volume, "axes_at", lambda _r, _c: _NotAnAxes3D())

    assert volume._volume_motion(_event(inaxes=object(), x=10.0, y=10.0)) is True
    assert volume._view_angles is None


# ---------------------------------------------------------------------------
# The same measurement on two of the three axes
# ---------------------------------------------------------------------------

def test_a_repeated_measurement_leaves_the_sweep_with_no_depth_axis(
        volume, monkeypatch):
    """The pickers allow the repeat, so both handlers refuse it rather than raise."""
    volume.set_spec(GraphSpec(x="a", y="b"), immediate=True)
    volume.set_mode("3D", z_column="a")
    volume.render_now()
    volume._volume_drag = ("a", 4.0, "b", 4.0)
    monkeypatch.setattr(volume, "screen_to_volume",
                        lambda _event: ("a", 6.0, "b", 6.0))

    volume._show_volume_drag(_event(x=10.0, y=10.0))
    assert volume._ghost == []

    assert volume._gate_from_volume_drag(_event(x=10.0, y=10.0)) is None
