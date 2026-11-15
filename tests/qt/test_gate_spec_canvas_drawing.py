"""How the Gate Editor's flat canvas draws, hit-tests and drags gate_spec gates.

The distinction held throughout is between "not drawn" and "drawn wrong": an
outline in the wrong units, a highlight from the wrong columns, or a gate
grabbable where it is invisible are all worse than the absence, so each is
asserted as an exact absence next to the ordinary drawing it must not swallow.
Decoration (log axes, colour ramps, binning) that cannot apply falls back
without taking the plot with it.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture; every case builds its own fresh canvas.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

from spacr.qt.widgets.gate_editor import GateCanvas  # noqa: E402
from spacr.qt.widgets.gate_settings import GateEditorSettings  # noqa: E402
from spacr.qt.widgets.gate_spec import (  # noqa: E402
    GATE_KINDS, POLYGON, RECTANGLE, BoxGate, CylinderGate, EllipseGate,
    GateError, GateSet, PolygonGate, PrismGate, RectGate, ThresholdGate,
)
from spacr.qt.widgets.graph_spec import GraphSpec  # noqa: E402

pytestmark = pytest.mark.qt

PALETTE = {"bg": "#000000"}


def _frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "a": rng.normal(5.0, 1.0, 200),
        "b": rng.normal(5.0, 1.0, 200),
        "c": rng.normal(5.0, 1.0, 200),
        "flat": np.zeros(200),
        "text": ["x"] * 200,
    })


@pytest.fixture
def make(qtbot):
    """Build a fresh canvas showing ``x`` against ``y`` of ``frame``."""
    def build(frame=None, x="a", y="b", gates=None, mode=None):
        canvas = GateCanvas()
        qtbot.addWidget(canvas)
        canvas.set_frame(_frame() if frame is None else frame)
        canvas.set_spec(GraphSpec(x=x, y=y))
        if mode is not None:
            canvas.set_mode(mode, z_column="c")
            canvas.set_anchor_axis("z")
        if gates is not None:
            canvas.set_gates(gates)
        return canvas
    return build


def _rect(name="box", x="a", y="b", low=4.0, high=6.0):
    return RectGate(name=name, x_column=x, y_column=y,
                    x_low=low, x_high=high, y_low=low, y_high=high)


def _axes(canvas):
    return list(canvas.panel_axes().values())[0]


def _dashed(ax):
    return [line for line in ax.lines if line.get_linestyle() == "--"]


def _outlines(ax):
    return [p for p in ax.patches if isinstance(p, MplPolygon)]


class _Mouse:
    """A press/motion carrying both data and pixel coordinates."""

    def __init__(self, ax, x_data, y_data):
        self.inaxes = ax
        self.xdata, self.ydata = x_data, y_data
        self.x, self.y = ax.transData.transform((x_data, y_data))
        self.button = 1
        self.step = 0
        self.key = None
        self.dblclick = False


class _Raising:
    """A transform that refuses, as a view mid-rotation can."""

    def transform(self, _point):
        raise ValueError("this transform is not invertible here")


def test_the_armed_tool_and_its_saved_default(make, subtests):
    """Only gate kinds arm; a saved default arms only an unarmed canvas."""
    with subtests.test(msg="unknown_tool_refused_and_keeps_the_armed_one"):
        canvas = make()
        before = canvas.tool
        assert before in GATE_KINDS
        with pytest.raises(GateError, match="unknown gate tool"):
            canvas.set_tool("lasso")
        assert canvas.tool == before
        canvas.set_tool("polygon")
        assert canvas.tool == "polygon"
        canvas.set_tool("")
        assert canvas.tool == ""
    cases = [
        ("namespace_default_arms_unarmed", "", types.SimpleNamespace(default_tool="polygon"), "polygon"),
        ("namespace_default_keeps_live_tool", "rectangle", types.SimpleNamespace(default_tool="polygon"), "rectangle"),
        ("settings_default_arms_unarmed", "", GateEditorSettings().replaced(default_tool=POLYGON), POLYGON),
        ("settings_default_keeps_live_tool", POLYGON, GateEditorSettings().replaced(default_tool=RECTANGLE), POLYGON),
    ]
    for case_id, armed, settings, want in cases:
        with subtests.test(msg=case_id):
            canvas = make()
            canvas.set_tool(armed)
            canvas.apply_settings(settings)
            assert canvas.tool == want


def test_decoration_that_cannot_apply_falls_back_without_losing_the_plot(make, monkeypatch, subtests):
    """Log scales, colour ramps, densities, binning and voxels refuse exactly."""
    ramp = np.linspace(0.0, 9.0, 60)
    log_cases = [
        ("log_over_zero_is_skipped", ramp, "linear"),
        ("log_over_positive_is_applied", ramp + 1.0, "log"),
    ]
    for case_id, values, want in log_cases:
        with subtests.test(msg=case_id):
            canvas = make(pd.DataFrame({"a": values, "b": values + 1.0}))
            canvas.apply_settings(GateEditorSettings().replaced(x_scale="log"))
            assert _axes(canvas).get_xscale() == want
            assert len(_axes(canvas).collections) == 2
    with subtests.test(msg="decorate_skips_log_over_a_zero_column"):
        canvas = make(x="flat")
        canvas._x_scale = "log"
        ax = canvas._figure.axes[0]
        assert canvas.decorate_axes(ax) is None
        assert ax.get_xscale() == "linear"
    for case_id, scale in (("refused_symlog", "symlog"), ("refused_log", "log")):
        with subtests.test(msg=case_id):
            canvas = make()
            ax = _axes(canvas)
            attempted = []

            def refuse(name, _attempted=attempted):
                _attempted.append(name)
                raise ValueError("unknown scale")

            with monkeypatch.context() as mp:
                mp.setattr(ax, "set_xscale", refuse)
                canvas._x_scale = scale
                assert canvas.decorate_axes(ax) is None
            assert attempted == [scale]
            assert ax.get_xscale() == "linear"
            assert len(ax.collections) == 2
    with subtests.test(msg="column_is_positive"):
        canvas = make()
        assert {c: canvas._column_is_positive(c)
                for c in ("nowhere", "flat", "text", "a")} == {
            "nowhere": False, "flat": False, "text": False, "a": True}
    with subtests.test(msg="colour_values"):
        canvas = make()
        rows = canvas.population()
        x, y = rows["a"].to_numpy(float), rows["b"].to_numpy(float)
        density = canvas._density(x, y)
        assert len(np.unique(density)) > 1
        for choice, want in (("text", density), ("density", density),
                             ("nowhere", density), ("c", rows["c"].to_numpy(float))):
            canvas._colour_by = choice
            np.testing.assert_array_equal(canvas._colour_values(x, y, rows), want)
        canvas._colour_by = "flat"
        assert canvas._colour_values(np.zeros(3), np.zeros(3), rows) is None
    with subtests.test(msg="density_over_nothing_finite_is_zero_per_point"):
        canvas = make()
        nan = np.full(5, np.nan)
        assert canvas._density(nan, nan).tolist() == [0.0] * 5
        assert canvas._density(np.array([np.nan, np.nan]),
                               np.array([np.nan, 1.0])).tolist() == [0.0, 0.0]
    with subtests.test(msg="binning"):
        canvas = make()
        assert canvas._draw_binned(_axes(canvas), np.array([]), np.array([])) is None
        canvas._resolution = "hexbin"
        canvas._bins = 40
        ax = Figure().add_subplot(111)
        rng = np.random.default_rng(1)
        artist = canvas._draw_binned(ax, rng.normal(size=500), rng.normal(size=500))
        assert list(ax.collections) == [artist]
    with subtests.test(msg="voxels_fall_back_to_dots"):
        canvas = make(mode="3D")
        canvas.apply_settings(GateEditorSettings().replaced(voxel_bins=8))
        n = canvas.VOXEL_THRESHOLD
        assert n == 20000
        small = np.linspace(0.0, 1.0, 10)
        assert canvas._draw_voxels(None, small, small, small) is False
        nan = np.full(n, np.nan)
        assert canvas._draw_voxels(None, nan, nan, nan) is False
        ax = Figure().add_subplot(111, projection="3d")
        values = np.zeros(n + 10)
        with monkeypatch.context() as mp:
            mp.setattr(np, "histogramdd", lambda *_a, **_k: (
                np.zeros((8, 8, 8)), [np.arange(9.0)] * 3))
            assert canvas._draw_voxels(ax, values, values, values) is False

            def refuse(*_a, **_k):
                raise MemoryError("that grid does not fit")

            mp.setattr(np, "histogramdd", refuse)
            assert canvas._draw_voxels(ax, values, values, values) is False
        assert canvas._draw_voxels(ax, values, values, values) is True
        assert len(ax.collections) == 1
        canvas._settings = types.SimpleNamespace(voxel_bins=1)
        assert canvas._draw_voxels(ax, values, values, values) is False


def test_a_gate_is_outlined_highlighted_and_previewed_only_on_its_own_axes(make, monkeypatch, subtests):
    """Outlines, rings and dashed placeholders appear on matching axes only."""
    outline_cases = [
        ("threshold_unplotted", ThresholdGate(name="t1", column="c", low=4.0), 0, 0),
        ("threshold_one_bound", ThresholdGate(name="t1", column="a", low=4.0), 1, 0),
        ("threshold_two_bounds", ThresholdGate(name="t1", column="a", low=4.0, high=6.0), 2, 0),
        ("rect_other_pair", _rect(x="a", y="c"), 0, 0),
        ("rect_these_axes", _rect(name="r1"), 0, 1),
        ("unknown_shape", types.SimpleNamespace(name="odd", x_column="a", y_column="b"), 0, 0),
    ]
    for case_id, gate, lines, patches in outline_cases:
        with subtests.test(msg=case_id):
            canvas = make()
            ax = Figure().add_subplot(111)
            canvas._artists = []
            canvas._outline(ax, gate, None)
            assert (len(ax.lines), len(_outlines(ax))) == (lines, patches)
            if lines + patches == 0:
                assert canvas._artists == []
            if case_id == "rect_these_axes":
                assert [t.get_text() for t in ax.texts] == ["r1"]
                assert canvas._gate_points(ax, gate) == [
                    (4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)]
    with subtests.test(msg="unknown_shape_has_no_points"):
        unknown = types.SimpleNamespace(name="odd", x_column="a", y_column="b")
        assert make()._gate_points(Figure().add_subplot(111), unknown) == []
    with subtests.test(msg="an_oval_is_a_64_point_polygon"):
        oval = EllipseGate(name="oval", x_column="a", y_column="b",
                           x_centre=5.0, y_centre=5.0, x_radius=1.0, y_radius=0.5)
        points = make()._gate_points(Figure().add_subplot(111), oval)
        assert len(points) == 64
        assert min(p[0] for p in points) == pytest.approx(4.0, abs=0.05)
        assert max(p[1] for p in points) == pytest.approx(5.5, abs=0.05)
    set_cases = [
        ("set_threshold_unplotted", ThresholdGate(name="t1", column="c", low=4.0), 0, 0),
        ("set_threshold_plotted", ThresholdGate(name="t1", column="a", low=4.0), 1, 0),
        ("set_rect_other_pair", _rect(x="a", y="c"), 0, 0),
    ]
    for case_id, gate, dashed, patches in set_cases:
        with subtests.test(msg=case_id):
            canvas = make(gates=GateSet().add(gate))
            assert (len(_dashed(_axes(canvas))), len(_outlines(_axes(canvas)))) == (dashed, patches)
    with subtests.test(msg="no_drawable_points_no_patch"):
        canvas = make()
        with monkeypatch.context() as mp:
            mp.setattr(canvas, "_gate_points", lambda ax, gate: [])
            canvas.set_gates(GateSet().add(_rect()))
            assert _outlines(_axes(canvas)) == []
    with subtests.test(msg="highlight_from_absent_columns_rings_nothing"):
        canvas = make()
        canvas._artists = []
        canvas._highlight(Figure().add_subplot(111), _rect(),
                          pd.DataFrame({"c": [1.0, 2.0]}), PALETTE)
        assert canvas._artists == []
    with subtests.test(msg="highlight_that_will_not_evaluate_rings_nothing"):
        gate = _rect()
        canvas = make(gates=GateSet((gate,)))
        canvas._artists = []

        def refuse(_frame, _name):
            raise ValueError("this gate cannot be evaluated here")

        with monkeypatch.context() as mp:
            mp.setattr(canvas._gates, "mask", refuse)
            canvas._highlight(Figure().add_subplot(111), gate,
                              canvas.population(), PALETTE)
        assert canvas._artists == []
    ghost_cases = [
        ("rect_ghost_then_none", _rect(), 1),
        ("threshold_ghost_is_its_lines", ThresholdGate(name="cut", column="a", low=4.0, high=6.0), 2),
        ("unknown_shape_ghost", types.SimpleNamespace(name="odd", x_column="a", y_column="b"), 0),
        ("cylinder_has_no_flat_ghost",
         CylinderGate(name="tube", u_column="a", v_column="b", axis_column="c",
                      u_centre=5.0, v_centre=5.0, u_radius=2.0, v_radius=2.0), 0),
    ]
    for case_id, gate, n in ghost_cases:
        with subtests.test(msg=case_id):
            canvas = make()
            canvas.render_now()
            canvas._show_ghost(gate)
            assert len(canvas._ghost) == n
            canvas._show_ghost(None)
            assert canvas._ghost == []
    with subtests.test(msg="a_ghost_already_gone_is_dropped_quietly"):
        class _Gone:
            def remove(self):
                raise ValueError("artist is not in the figure")

        canvas = make()
        canvas._ghost = [_Gone()]
        canvas._clear_ghost()
        assert canvas._ghost == []
    drag_cases = [
        ("threshold_both_bounds_follow", ThresholdGate(name="cut", column="a", low=2.0, high=6.0),
         (4.0, 5.0), [3.0, 7.0]),
        ("open_ended_cut_shows_one_line", ThresholdGate(name="cut", column="a", low=2.0, high=None),
         (6.0, 7.0), [3.0]),
    ]
    for case_id, gate, (start, end), want in drag_cases:
        with subtests.test(msg=case_id):
            frame = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0]})
            canvas = make(frame, gates=GateSet().add(gate))
            ax = canvas.axes_at(0, 0)
            canvas._on_press(_Mouse(ax, start, 5.0))
            assert canvas._move_name == "cut"
            canvas._on_motion(_Mouse(ax, end, 5.0))
            assert sorted(float(line.get_xdata()[0]) for line in canvas._ghost) == want


def test_a_gate_is_grabbed_and_dragged_only_where_it_is_drawn(make, monkeypatch, subtests):
    """Handles, hit-tests and drags: a gate off these axes can never be picked up."""
    class _Nameless(ThresholdGate):
        @property
        def columns(self):
            return ()

    class _Unanchorable(RectGate):
        def handles(self, view):
            raise RuntimeError("no view box to place anchors in")

    class _Unprobeable(RectGate):
        def mask(self, frame):
            raise RuntimeError("this gate cannot be probed")

    class _ColumnlessBox(BoxGate):
        @property
        def columns(self):
            return ()

    with subtests.test(msg="on_these_axes"):
        canvas = make()
        assert canvas._gate_is_on_these_axes(_rect()) is True
        assert canvas._gate_is_on_these_axes(_rect(x="b", y="c")) is False
        assert canvas._gate_is_on_these_axes(types.SimpleNamespace(name="odd", columns=())) is False
        assert canvas._gate_is_on_these_axes(_Nameless(name="t1", column="a", low=4.0)) is False
    handle_cases = [
        ("rect_here", _rect(), ["x_low,y_low", "x_high,y_low", "x_low,y_high", "x_high,y_high"]),
        ("rect_elsewhere", _rect(x="b", y="c"), []),
        ("unanchorable", _Unanchorable(name="r1", x_column="a", y_column="b",
                                       x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0), []),
    ]
    for case_id, gate, corners in handle_cases:
        with subtests.test(msg=case_id):
            canvas = make()
            ax = _axes(canvas)
            handles = canvas._handles_for(ax, gate)
            assert [h.role for h in handles if h.corner] == corners
            before = (len(ax.lines), len(ax.collections), len(canvas._artists))
            canvas._draw_handles(ax, gate, PALETTE)
            after = (len(ax.lines), len(ax.collections), len(canvas._artists))
            assert (after == before) is (not handles)
    with subtests.test(msg="handles_that_raise_on_the_class"):
        gate = _rect()
        canvas = make()
        with monkeypatch.context() as mp:
            mp.setattr(type(gate), "handles", lambda _self, _view: (_ for _ in ()).throw(
                RuntimeError("cannot place handles on an unbounded side")))
            assert canvas._handles_for(Figure().add_subplot(111), gate) == ()
            canvas._artists = []
            canvas._draw_handles(Figure().add_subplot(111), gate, PALETTE)
            assert canvas._artists == []

    def gates_of(*gates):
        out = GateSet()
        out.gates.extend(gates)
        return out

    box3 = BoxGate(name="volume-box", x_column="a", y_column="b", z_column="c",
                   x_low=0.0, x_high=10.0, y_low=0.0, y_high=10.0, z_low=0.0, z_high=10.0)
    hit_cases = [
        ("ordinary_hit", gates_of(_rect(name="good")), (5.0, 5.0), "good"),
        ("ordinary_miss", gates_of(_rect(name="good")), (100.0, 100.0), None),
        ("other_pair", gates_of(_rect(x="b", y="c")), (5.0, 5.0), None),
        ("nameless", gates_of(_Nameless(name="t1", column="a", low=4.0)), (5.0, 5.0), None),
        ("broken_mask_skipped",
         gates_of(_Unprobeable(name="bad", x_column="a", y_column="b", x_low=4.0,
                               x_high=6.0, y_low=4.0, y_high=6.0), _rect(name="good")),
         (5.0, 5.0), "good"),
        ("box_over_rect_seen_from_front", gates_of(box3, _rect(name="flat", low=2.0, high=8.0)),
         (5.0, 5.0), "flat"),
        ("columnless_box", gates_of(_ColumnlessBox(name="b1", x_column="a", y_column="b",
                                                   z_column="c", x_low=4.0, x_high=6.0,
                                                   y_low=4.0, y_high=6.0)), (5.0, 5.0), None),
    ]
    two_columns = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0]})
    for case_id, gates, (x, y), want in hit_cases:
        with subtests.test(msg=case_id):
            canvas = make(two_columns if case_id.startswith("box_over") else None, gates=gates)
            canvas.render_now()
            assert canvas.gate_at(x, y) == want
    with subtests.test(msg="columnless_box_is_on_these_axes"):
        canvas = make()
        assert canvas._gate_is_on_these_axes(_ColumnlessBox(
            name="b1", x_column="a", y_column="b", z_column="c",
            x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)) is True
    with subtests.test(msg="mask_raising_on_the_class_unpicks_it"):
        good = _rect(name="good")
        canvas = make(gates=GateSet((good,)))
        assert canvas.gate_at(5.0, 5.0) == "good"
        with monkeypatch.context() as mp:
            mp.setattr(type(good), "mask", lambda _self, _frame: (_ for _ in ()).throw(
                ValueError("this gate cannot be evaluated on that probe")))
            assert canvas.gate_at(5.0, 5.0) is None
    with subtests.test(msg="press_on_a_gate_elsewhere_moves_nothing"):
        frame = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0]})
        canvas = make(frame, gates=GateSet().add(RectGate(
            name="elsewhere", x_column="p", y_column="q",
            x_low=0.0, x_high=10.0, y_low=0.0, y_high=10.0)))
        ax = canvas.axes_at(0, 0)
        assert canvas.gate_at(5.0, 5.0) is None
        canvas._on_press(_Mouse(ax, 5.0, 5.0))
        assert canvas._move_name is None
    with subtests.test(msg="handle_at_refusals"):
        canvas = make(gates=GateSet((_rect(),)))
        canvas.render_now()
        ax = canvas._figure.axes[0]
        assert canvas.handle_at(types.SimpleNamespace(inaxes=None, x=1, y=1)) is None
        assert canvas.handle_at(types.SimpleNamespace(inaxes=ax, x=None, y=None)) is None
        real = ax.transData
        ax.transData = _Raising()
        try:
            assert canvas.handle_at(types.SimpleNamespace(inaxes=ax, x=10.0, y=10.0)) is None
        finally:
            ax.transData = real
    drag_cases = [
        ("never_reached_the_axes", None, "", None, dict(inaxes=None, xdata=None, ydata=None)),
        ("nothing_picked_up", None, "", None, dict(xdata=5.0, ydata=5.0)),
        ("refused_role", ("box", "not_a_handle"), "", None, dict(xdata=5.0, ydata=5.0)),
        ("gate_that_is_gone", None, "deleted", (5.0, 5.0), dict(xdata=6.0, ydata=6.0)),
        ("non_finite_pointer", ("box", "x_high"), "", None, dict(xdata=float("nan"), ydata=0.0)),
    ]
    for case_id, resize, move_name, move_from, event in drag_cases:
        with subtests.test(msg=case_id):
            canvas = make(gates=GateSet((_rect(),)))
            canvas.render_now()
            ax = canvas._figure.axes[0]
            canvas._resize, canvas._move_name, canvas._move_from = resize, move_name, move_from
            fields = dict(inaxes=ax, x=0.0, y=0.0)
            fields.update(event)
            assert canvas._dragged_to(types.SimpleNamespace(**fields)) is None
    with subtests.test(msg="finite_pointer_resizes_one_side"):
        canvas = make(gates=GateSet((_rect(),)))
        canvas.render_now()
        canvas._resize = ("box", "x_high")
        pulled = canvas._dragged_to(types.SimpleNamespace(
            inaxes=canvas._figure.axes[0], x=0.0, y=0.0, xdata=8.0, ydata=0.0))
        assert (pulled.x_low, pulled.x_high, pulled.y_low, pulled.y_high) == (4.0, 8.0, 4.0, 6.0)
    with subtests.test(msg="flat_box_is_picked_and_resized_keeping_its_depth"):
        frame = pd.DataFrame({"a": [0.0, 5.0, 10.0], "b": [0.0, 5.0, 10.0], "c": [0.0, 5.0, 10.0]})
        box = BoxGate(name="g", x_column="a", y_column="b", z_column="c",
                      x_low=2.0, x_high=8.0, y_low=2.0, y_high=8.0, z_low=2.0, z_high=8.0)
        canvas = make(frame, gates=GateSet().add(box))
        ax = canvas.axes_at(0, 0)
        assert canvas.gate_at(5.0, 5.0) == "g"
        assert canvas.gate_at(9.5, 9.5) is None
        canvas._resize = ("g", "x_low,y_low")
        pulled = canvas._dragged_to(_Mouse(ax, 4.0, 4.0))
        assert isinstance(pulled, BoxGate)
        assert (pulled.x_low, pulled.x_high, pulled.y_low, pulled.y_high,
                pulled.z_low, pulled.z_high) == (4.0, 8.0, 4.0, 8.0, 2.0, 8.0)
    with subtests.test(msg="cylinder_footprint_edit_keeps_its_axis_range"):
        gate = CylinderGate(name="cyl", u_column="a", v_column="b", axis_column="c",
                            u_radius=1, v_radius=1, axis_low=-.25, axis_high=.5)
        canvas = make(gates=GateSet([gate]))
        footprint = PolygonGate(name="cyl", x_column="a", y_column="b",
                                vertices=((-1, -1), (1, -1), (1, 1), (-1, 1)))
        canvas._as_flat = lambda shown: footprint if shown is gate else shown
        canvas._resize = ("cyl", "outline:1:0.5")
        edited = canvas._dragged_to(types.SimpleNamespace(
            inaxes=canvas.axes_at(0, 0), x=0.0, y=0.0, xdata=1.5, ydata=0.0))
        assert isinstance(edited, PrismGate)
        assert edited.vertices == ((-1, -1), (1, -1), (1.5, 0), (1, 1), (-1, 1))
        assert (edited.axis_column, edited.axis_low, edited.axis_high) == ("c", -.25, .5)
