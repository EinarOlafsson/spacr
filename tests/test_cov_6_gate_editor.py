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
from spacr.qt.widgets.gate_settings import GateEditorSettings  # noqa: E402
from spacr.qt.widgets.gate_spec import (  # noqa: E402
    BoxGate, GateSet, RectGate, ThresholdGate,
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


def test_a_default_tool_from_the_settings_arms_an_unarmed_canvas(canvas):
    """A saved workspace re-arms the tool the user was last drawing with."""
    from spacr.qt.widgets.gate_spec import POLYGON

    canvas.set_tool("")
    canvas.apply_settings(
        GateEditorSettings().replaced(default_tool=POLYGON))
    assert canvas.tool == POLYGON


def test_a_default_tool_does_not_override_one_already_armed(canvas):
    """Re-applying settings mid-draw must not swap the tool under the user."""
    from spacr.qt.widgets.gate_spec import POLYGON, RECTANGLE

    canvas.set_tool(POLYGON)
    canvas.apply_settings(
        GateEditorSettings().replaced(default_tool=RECTANGLE))
    assert canvas.tool == POLYGON


# ---------------------------------------------------------------------------
# Axis scales
# ---------------------------------------------------------------------------

def test_a_log_axis_over_a_column_that_reaches_zero_is_not_applied(qtbot):
    """A log axis over non-positive data draws nothing, which reads as broken."""
    widget = GateCanvas()
    qtbot.addWidget(widget)
    values = np.linspace(0.0, 9.0, 60)
    widget.set_frame(pd.DataFrame({"a": values, "b": values + 1.0}))
    widget.set_spec(GraphSpec(x="a", y="b"))
    widget.apply_settings(GateEditorSettings().replaced(x_scale="log"))
    assert _axes(widget).get_xscale() == "linear"
    assert _axes(widget).collections, "the plot went away"


def test_a_log_axis_over_strictly_positive_data_is_applied(qtbot):
    """The refusal above is about the data, not about the setting."""
    widget = GateCanvas()
    qtbot.addWidget(widget)
    values = np.linspace(1.0, 10.0, 60)
    widget.set_frame(pd.DataFrame({"a": values, "b": values}))
    widget.set_spec(GraphSpec(x="a", y="b"))
    widget.apply_settings(GateEditorSettings().replaced(x_scale="log"))
    assert _axes(widget).get_xscale() == "log"


def test_a_column_that_is_not_on_the_table_is_not_positive(canvas):
    """"cannot tell" has to answer the same way as "reaches zero"."""
    assert canvas._column_is_positive("no_such_column") is False
    assert canvas._column_is_positive("a") is True


def test_a_scale_matplotlib_refuses_does_not_take_the_plot_with_it(canvas,
                                                                   monkeypatch):
    """A scale name can be rejected by the backend; the points still draw."""
    axes = _axes(canvas)

    def refuse(*args, **kwargs):
        raise ValueError("unrecognised scale")

    monkeypatch.setattr(axes, "set_xscale", refuse)
    canvas._x_scale = "log"
    canvas.decorate_axes(axes)
    assert axes.get_xscale() == "linear"
    assert axes.collections, "the plot went away with the scale"


# ---------------------------------------------------------------------------
# Colouring
# ---------------------------------------------------------------------------

def test_a_colour_column_with_no_numbers_in_it_falls_back_to_density(canvas,
                                                                     frame):
    """The colour axis is decoration; an error there would lose the plot."""
    canvas._colour_by = "label"
    values = canvas._colour_values(
        frame["a"].to_numpy(float), frame["b"].to_numpy(float), frame)
    assert values is not None
    assert len(np.unique(values)) > 1, "density varies; a flat fill does not"


def test_a_numeric_colour_column_wins_over_density(canvas, frame):
    """"colour by c" must mean that and not an approximation of it."""
    canvas._colour_by = "c"
    values = canvas._colour_values(
        frame["a"].to_numpy(float), frame["b"].to_numpy(float), frame)
    assert np.allclose(values, frame["c"].to_numpy(float))


def test_a_flat_colour_asks_for_no_values_at_all(canvas, frame):
    """``None`` is what tells the scatter to use one colour."""
    canvas._colour_by = "flat"
    assert canvas._colour_values(np.zeros(3), np.zeros(3), frame) is None


# ---------------------------------------------------------------------------
# Binned drawing and density
# ---------------------------------------------------------------------------

def test_binning_nothing_draws_nothing(canvas):
    """A filter that kept no rows must not reach numpy's histogram."""
    assert canvas._draw_binned(_axes(canvas), np.array([]), np.array([])) is None


def test_a_density_over_no_finite_points_is_all_zero(canvas):
    """Every point gets a value, so the array has to keep its length."""
    values = canvas._density(np.array([np.nan, np.nan]),
                             np.array([np.nan, 1.0]))
    assert values.tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Voxels
# ---------------------------------------------------------------------------

def test_a_volume_of_nothing_measurable_falls_back_to_the_scatter(canvas):
    """No occupied bin means no voxel grid, and the caller draws points.

    ``histogramdd`` puts non-finite coordinates in no bin at all, so a
    volume whose third measurement was never computed produces an empty
    grid rather than a wrong one.
    """
    canvas.apply_settings(GateEditorSettings().replaced(voxel_bins=8))
    count = canvas.VOXEL_THRESHOLD
    nan = np.full(count, np.nan)
    assert canvas._draw_voxels(None, nan, nan, nan) is False


def test_too_few_points_do_not_justify_binning(canvas):
    """The threshold is what keeps a small volume drawn as real objects."""
    canvas.apply_settings(GateEditorSettings().replaced(voxel_bins=8))
    small = np.linspace(0.0, 1.0, 10)
    assert canvas._draw_voxels(None, small, small, small) is False


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


def test_a_gate_naming_no_measurement_is_on_no_axes(canvas):
    """A gate with no columns cannot be tested against the ones on screen."""

    class _Nameless(ThresholdGate):
        @property
        def columns(self):
            return ()

    gate = _Nameless(name="t1", column="a", low=4.0)
    assert canvas._gate_is_on_these_axes(gate) is False


def test_a_threshold_on_a_measurement_that_is_not_plotted_is_not_drawn(canvas):
    """A line at 4.0 on axes measuring something else is a false statement."""
    gates = GateSet().add(ThresholdGate(name="t1", column="c", low=4.0))
    canvas.set_gates(gates)
    axes = _axes(canvas)
    assert not [line for line in axes.lines
                if line.get_linestyle() == "--"], "an unrelated cut was drawn"


def test_a_threshold_on_a_plotted_measurement_is_drawn(canvas):
    """The refusal above must not be swallowing the ordinary outline."""
    gates = GateSet().add(ThresholdGate(name="t1", column="a", low=4.0))
    canvas.set_gates(gates)
    assert [line for line in _axes(canvas).lines
            if line.get_linestyle() == "--"]


def test_a_rectangle_on_another_pair_of_measurements_is_not_outlined(canvas):
    """An outline in the wrong units is worse than no outline."""
    from matplotlib.patches import Polygon as MplPolygon

    canvas.set_gates(GateSet().add(_rect(x="a", y="c")))
    assert not [p for p in _axes(canvas).patches
                if isinstance(p, MplPolygon)]


def test_a_polygon_with_no_drawable_points_is_not_outlined(canvas,
                                                           monkeypatch):
    """A shape that projects to nothing must not become an empty patch."""
    from matplotlib.patches import Polygon as MplPolygon

    monkeypatch.setattr(canvas, "_gate_points", lambda ax, gate: [])
    canvas.set_gates(GateSet().add(_rect()))
    assert not [p for p in _axes(canvas).patches
                if isinstance(p, MplPolygon)]


# ---------------------------------------------------------------------------
# Handles
# ---------------------------------------------------------------------------

class _UnanchorableRect(RectGate):
    """A rectangle whose anchor points cannot be computed."""

    def handles(self, view):
        raise RuntimeError("no view box to place anchors in")


def test_a_gate_whose_handles_cannot_be_computed_gets_none(canvas):
    """Anchors are for dragging; failing to place them must not lose the gate."""
    gate = _UnanchorableRect(name="r1", x_column="a", y_column="b",
                             x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)
    assert canvas._handles_for(_axes(canvas), gate) == ()


def test_a_gate_on_these_axes_does_offer_handles(canvas):
    """The refusal above must not be swallowing the ordinary anchors."""
    assert canvas._handles_for(_axes(canvas), _rect()) != ()


def test_a_gate_off_these_axes_offers_no_handles(canvas):
    """Grabbing an invisible gate is how a plot appears to move on its own."""
    assert canvas._handles_for(_axes(canvas), _rect(x="a", y="c")) == ()


def test_drawing_handles_for_a_gate_that_has_none_is_a_no_op(canvas):
    """Called for every gate on every redraw, including the ones off-axes."""
    axes = _axes(canvas)
    before = len(axes.lines), len(axes.collections), len(canvas._artists)
    canvas._draw_handles(axes, _rect(x="a", y="c"), _palette())
    assert (len(axes.lines), len(axes.collections),
            len(canvas._artists)) == before


# ---------------------------------------------------------------------------
# Hit-testing
# ---------------------------------------------------------------------------

def test_a_gate_naming_no_measurement_cannot_be_clicked(canvas):
    """Hit-testing is pure geometry, and there is no geometry without columns."""

    class _Nameless(ThresholdGate):
        @property
        def columns(self):
            return ()

    gates = GateSet()
    gates.gates.append(_Nameless(name="t1", column="a", low=4.0))
    canvas.set_gates(gates)
    assert canvas.gate_at(5.0, 5.0) is None


class _UnprobeableRect(RectGate):
    """A rectangle that cannot be hit-tested at all."""

    def mask(self, frame):
        raise RuntimeError("this gate cannot be probed")


def test_a_gate_whose_mask_raises_does_not_stop_the_ones_that_can_be_hit(
        canvas):
    """One broken gate must not make the whole plot unclickable."""
    bad = _UnprobeableRect(name="bad", x_column="a", y_column="b",
                            x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)
    gates = GateSet()
    gates.gates.extend([bad, _rect(name="good")])
    canvas.set_gates(gates)
    assert canvas.gate_at(5.0, 5.0) == "good"


def test_a_click_outside_every_gate_hits_nothing(canvas):
    """The hits above must not be swallowing the ordinary miss."""
    canvas.set_gates(GateSet().add(_rect()))
    assert canvas.gate_at(100.0, 100.0) is None


# ---------------------------------------------------------------------------
# Outlines drawn directly, so the "not on these axes" cases are observable
# ---------------------------------------------------------------------------

def _palette():
    from spacr.qt.theme import active_palette

    return active_palette()


def test_a_threshold_on_an_unplotted_measurement_adds_no_line(canvas):
    """A dashed cut at 4.0 on axes measuring something else is a false claim."""
    axes = _axes(canvas)
    before = len(axes.lines)
    canvas._outline(axes, ThresholdGate(name="t1", column="c", low=4.0),
                    _palette())
    assert len(axes.lines) == before


def test_a_threshold_on_a_plotted_measurement_adds_one_line_per_bound(canvas):
    """The refusal above must not be swallowing the ordinary outline."""
    axes = _axes(canvas)
    before = len(axes.lines)
    canvas._outline(axes, ThresholdGate(name="t1", column="a", low=4.0,
                                        high=6.0), _palette())
    assert len(axes.lines) == before + 2


def test_a_rectangle_on_another_pair_adds_no_patch(canvas):
    """An outline in the wrong units is worse than no outline."""
    from matplotlib.patches import Polygon as MplPolygon

    axes = _axes(canvas)
    before = [p for p in axes.patches if isinstance(p, MplPolygon)]
    canvas._outline(axes, _rect(x="a", y="c"), _palette())
    after = [p for p in axes.patches if isinstance(p, MplPolygon)]
    assert len(after) == len(before)


def test_a_rectangle_on_these_axes_is_outlined_and_named(canvas):
    """The refusal above must not be swallowing the ordinary rectangle."""
    from matplotlib.patches import Polygon as MplPolygon

    axes = _axes(canvas)
    canvas._outline(axes, _rect(name="r1"), _palette())
    patches = [p for p in axes.patches if isinstance(p, MplPolygon)]
    assert patches
    assert any(text.get_text() == "r1" for text in axes.texts)


# ---------------------------------------------------------------------------
# Hit-testing a shape that names no measurement
# ---------------------------------------------------------------------------

class _ColumnlessBox(BoxGate):
    """A box on these axes that reports no columns to probe with."""

    @property
    def columns(self):
        return ()


def test_a_shape_on_these_axes_that_names_no_column_cannot_be_probed(canvas):
    """Hit-testing builds the probe frame FROM the columns.

    A box is judged to be on these axes by its ``x_column``/``y_column``
    alone, because that is what its flat rectangle is drawn from. With no
    columns to probe with there is no frame to build and no geometry to
    test, so it is skipped rather than reported as a hit at every click.
    """
    box = _ColumnlessBox(name="b1", x_column="a", y_column="b", z_column="c",
                         x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)
    gates = GateSet()
    gates.gates.append(box)
    canvas.set_gates(gates)
    assert canvas._gate_is_on_these_axes(box) is True
    assert canvas.gate_at(5.0, 5.0) is None


# ---------------------------------------------------------------------------
# Closing a volume polygon
# ---------------------------------------------------------------------------

class _UnprojectableAxes:
    """3-D axes whose transform refuses, as one mid-rotation can."""

    class transData:
        @staticmethod
        def transform(point):
            raise RuntimeError("the projection is not set up yet")


def test_a_volume_click_that_cannot_be_projected_does_not_close_the_polygon(
        canvas, monkeypatch):
    """Closing on a failed projection would finish a shape somewhere else.

    ``Axes3D.transData`` is only valid once the view has been drawn. A click
    arriving before that has no pixel distance to compare, so the answer is
    "not near the first vertex" rather than a guess.
    """
    canvas._pending = [(1.0, 2.0)]
    canvas._pending_plane = ("a", "b")
    monkeypatch.setattr(canvas, "_volume_face_point",
                        lambda ax, u, v: (1.0, 2.0, 3.0))
    event = type("_Event", (), {"inaxes": _UnprojectableAxes(),
                                "x": 10, "y": 10})()
    assert canvas._near_first_volume_vertex(event) is False


def test_a_volume_click_with_no_axes_under_it_closes_nothing(canvas):
    """A click in the margin has no axes and therefore no vertex."""
    canvas._pending = [(1.0, 2.0)]
    canvas._pending_plane = ("a", "b")
    event = type("_Event", (), {"inaxes": None, "x": 10, "y": 10})()
    assert canvas._near_first_volume_vertex(event) is False


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
