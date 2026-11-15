"""How the Gate Editor's 3D volume turns gestures into gate_spec gates.

A volume gate is two gestures: a footprint on the picked plane, then a drag
along its normal for depth.  The volume also spins, zooms, sweeps footprints
and draws the gates where their numbers are.  Each gesture has to survive a
view it cannot project, a click that means "full depth", a depth the gate
refuses, and a measurement the plot does not show -- by refusing with a
sentence or doing nothing, never by inventing a number.

One general test per behaviour family, each walking a case table with pytest's
``subtests`` fixture; every case builds its own fresh canvas.
"""
from __future__ import annotations

import types
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

import spacr.qt.widgets.gate_editor as editor  # noqa: E402
from spacr.qt.widgets.gate_editor import GateCanvas  # noqa: E402
from spacr.qt.widgets.gate_spec import (  # noqa: E402
    BoxGate, CylinderGate, GateError, GateSet, RectGate, ThresholdGate,
    ViewGate, _EllipsoidGate,
)
from spacr.qt.widgets.graph_spec import GraphSpec  # noqa: E402
from tests.qt.test_gate_editor_3d_mode_is_usable import (  # noqa: E402,F401
    _Mouse, screen)

pytestmark = pytest.mark.qt


def _frame():
    rng = np.random.default_rng(0)
    return pd.DataFrame({"a": rng.normal(5.0, 1.0, 200),
                         "b": rng.normal(5.0, 1.0, 200),
                         "c": rng.normal(5.0, 1.0, 200)})


@pytest.fixture
def make(qtbot):
    """A fresh canvas: 3D over a/b/c anchored on z unless ``mode`` says 2D."""
    def build(mode="3D", x="a", y="b", z="c", render=True):
        canvas = GateCanvas()
        qtbot.addWidget(canvas)
        canvas.set_frame(_frame())
        canvas.set_spec(GraphSpec(x=x, y=y))
        if mode == "3D":
            canvas.set_mode("3D", z_column=z)
            canvas.set_anchor_axis("z")
        if render:
            canvas.render_now()
        return canvas
    return build


def _event(**kwargs):
    base = dict(inaxes=None, x=0.0, y=0.0, xdata=None, ydata=None)
    base.update(kwargs)
    return types.SimpleNamespace(**base)


def _rect():
    return RectGate(name="box", x_column="a", y_column="b",
                    x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)


def _footprint():
    return BoxGate(name="cube", x_column="a", y_column="b", z_column="c",
                   x_low=4.0, x_high=6.0, y_low=4.0, y_high=6.0)


class _Transform:
    """A transData answering from a list (raising if told to)."""

    def __init__(self, *answers, error=None):
        self._answers = list(answers)
        self._error = error
        self._n = 0

    def transform(self, _point):
        if self._error is not None:
            raise self._error
        answer = self._answers[min(self._n, len(self._answers) - 1)]
        self._n += 1
        return answer


def test_a_footprint_gets_its_depth_from_a_second_gesture(make, monkeypatch, subtests):
    """_begin/_finish_volume_depth and _depth_bounds_from_drag, exactly."""
    with subtests.test(msg="flat_view_finishes_at_once"):
        canvas = make(mode="2D")
        drawn = []
        canvas.gate_drawn.connect(drawn.append)
        for gate in (_rect(), BoxGate(name="(unnamed)", x_column="a",
                                      y_column="b", z_column="c")):
            canvas._begin_volume_depth(gate)
            assert drawn[-1] is gate
            assert canvas._pending_volume_gate is None
        assert canvas._depth_bounds_from_drag((0.0, 0.0), _event(x=50.0, y=50.0)) is None
    with subtests.test(msg="volume_asks_for_depth_then_finishes"):
        canvas = make()
        said, drawn = [], []
        canvas.depth_requested.connect(said.append)
        canvas.gate_drawn.connect(drawn.append)
        canvas._begin_volume_depth(_footprint())
        assert (canvas._pending_volume_gate, canvas._pending_volume_axis) == (_footprint(), "c")
        assert said == ["Footprint ready on a / b. Drag once more along c to "
                        "set its depth; click for full depth."]
        gate = canvas._finish_volume_depth((1.0, 8.0))
        assert (gate.z_low, gate.z_high, gate.x_low, gate.x_high) == (1.0, 8.0, 4.0, 6.0)
        assert drawn == [gate] and said[-1] == ""
        assert (canvas._pending_volume_gate, canvas._pending_volume_axis) == (None, "")
        assert canvas.pending_depth() == (None, None)
    with subtests.test(msg="a_finished_cylinder_clears_the_message"):
        canvas = make()
        said, drawn = [], []
        canvas.depth_requested.connect(said.append)
        canvas.gate_drawn.connect(drawn.append)
        canvas._pending_volume_gate = CylinderGate(
            name="(unnamed)", u_column="a", v_column="b", axis_column="c",
            u_centre=1.0, v_centre=1.0, u_radius=1.0, v_radius=1.0)
        canvas._pending_volume_axis = "c"
        gate = canvas._finish_volume_depth((0.0, 15.0))
        assert (gate.axis_low, gate.axis_high) == (0.0, 15.0)
        assert said == [""] and drawn == [gate]
    refusals = [
        ("nobody_started", None, "c", (0.0, 1.0), None),
        ("no_axis_named", _rect(), "", (0.0, 1.0), None),
        ("box_with_no_axis", BoxGate(name="(unnamed)", x_column="a", y_column="b",
                                     z_column="c"), "", (1.0, 2.0), None),
        ("gate_refuses_the_depth",
         types.SimpleNamespace(with_threshold=lambda *_a: (_ for _ in ()).throw(
             GateError("a gate needs a low below its high"))),
         "c", (5.0, 1.0), "a gate needs a low below its high"),
        ("threshold_has_no_side_for_it", ThresholdGate(name="(unnamed)", column="a",
                                                      low=0.0, high=1.0),
         "c", (1.0, 2.0), "no bound on 'c'"),
    ]
    for case_id, pending, axis, bounds, message in refusals:
        with subtests.test(msg=case_id):
            canvas = make()
            said, drawn = [], []
            canvas.depth_requested.connect(said.append)
            canvas.gate_drawn.connect(drawn.append)
            canvas._pending_volume_gate, canvas._pending_volume_axis = pending, axis
            assert canvas._finish_volume_depth(bounds) is None
            assert drawn == []
            assert canvas._pending_volume_axis == axis
            if message is None:
                assert said == []
            else:
                assert len(said) == 1 and message in said[0]
    with subtests.test(msg="pending_depth_given_backwards"):
        canvas = make()
        canvas.set_pending_depth(9.0, 2.0)
        assert canvas.pending_depth() == (2.0, 9.0)
    bound_cases = [
        ("click_is_full_depth", "c", None, None, (100.0, 100.0), (100.0, 100.0), (None, None)),
        ("axis_not_shown", "not_a_column", None, None, (0.0, 0.0), (50.0, 50.0), "none"),
        ("no_range", "c", None, (5.0, 5.0), (0.0, 0.0), (80.0, 80.0), "none"),
        ("projection_refuses", "c", _Transform(error=ValueError("x")), None,
         (0.0, 0.0), (80.0, 80.0), "none"),
        ("edge_on_normal", "c", _Transform((100.0, 100.0)), None,
         (0.0, 0.0), (180.0, 180.0), "none"),
        ("spans_whole_axis", "c", _Transform((0.0, 0.0), (0.0, 10.0)), None,
         (0.0, 0.0), (0.0, 100.0), (None, None)),
        ("partial_drag_is_a_slab", "c", _Transform((0.0, 0.0), (0.0, 100.0)), None,
         (0.0, 0.0), (0.0, 50.0), "half"),
    ]
    for case_id, axis, transform, zlim, start, end, want in bound_cases:
        with subtests.test(msg=case_id):
            canvas = make()
            canvas._pending_volume_axis = axis
            ax = canvas.axes_at(0, 0)
            low, high = (float(v) for v in ax.get_zlim3d())
            real = ax.transData
            with monkeypatch.context() as mp:
                if zlim is not None:
                    mp.setattr(ax, "get_zlim3d", lambda: zlim)
                if transform is not None:
                    ax.transData = transform
                try:
                    got = canvas._depth_bounds_from_drag(start, _event(x=end[0], y=end[1]))
                finally:
                    ax.transData = real
            if want == "none":
                assert got is None
            elif want == "half":
                assert got == pytest.approx((low, low + (high - low) * 0.5))
            else:
                assert got == want
    with subtests.test(msg="real_projection_drag_stays_inside_the_limits"):
        canvas = make()
        canvas._pending_volume_axis = "c"
        low, high = (float(v) for v in canvas.axes_at(0, 0).get_zlim3d())
        bounds = canvas._depth_bounds_from_drag((0.0, 0.0), _event(x=0.0, y=40.0))
        assert bounds == (None, None) or (
            bounds[0] == pytest.approx(low) and low < bounds[1] <= high)


def test_volume_gestures_spin_sweep_and_zoom_or_refuse(make, monkeypatch, subtests):
    """_volume_press/_motion/_release/_scroll and the sweep previews, exactly."""
    with subtests.test(msg="flat_view_leaves_every_event_alone"):
        canvas = make(mode="2D")
        assert [canvas._volume_press(_event()), canvas._volume_motion(_event()),
                canvas._volume_release(_event()), canvas._volume_scroll(_event())] == [False] * 4
    with subtests.test(msg="press_outside_the_axes_is_consumed"):
        canvas = make()
        assert canvas._volume_press(_event(inaxes=None)) is True
        assert canvas._volume_drag is None
    with subtests.test(msg="polygon_tool_takes_clicks"):
        canvas = make()
        canvas.set_drag_mode("draw")
        canvas.set_volume_shape("polygon")
        assert canvas._volume_press(_event(inaxes=canvas.axes_at(0, 0), x=10.0, y=10.0)) is False
    with subtests.test(msg="press_with_a_footprint_waiting_starts_depth"):
        canvas = make()
        canvas.set_drag_mode("draw")
        canvas.set_volume_shape("box")
        canvas._pending_volume_gate = _footprint()
        assert canvas._volume_press(_event(inaxes=canvas.axes_at(0, 0), x=12.0, y=34.0)) is True
        assert canvas._depth_drag_from == (12.0, 34.0)
    with subtests.test(msg="spin_about_the_locked_axis"):
        canvas = make()
        canvas.set_drag_mode("spin")
        ax = canvas.axes_at(0, 0)
        assert canvas._volume_press(_event(inaxes=ax, x=0.0, y=0.0)) is True
        before = (float(ax.elev), float(ax.azim))
        canvas.set_spin_axis("z")
        canvas._volume_motion(_event(inaxes=ax, x=40.0, y=0.0))
        assert float(ax.elev) == pytest.approx(before[0])
        assert float(ax.azim) != pytest.approx(before[1])
        canvas.set_spin_axis("x")
        canvas._volume_motion(_event(inaxes=ax, x=40.0, y=30.0))
        assert float(ax.elev) != pytest.approx(before[0])
        canvas.set_spin_axis("")
        turned = (float(ax.elev), float(ax.azim))
        canvas._volume_motion(_event(inaxes=ax, x=80.0, y=60.0))
        assert (float(ax.elev), float(ax.azim)) != turned
    with subtests.test(msg="motion_with_nothing_in_flight"):
        canvas = make()
        canvas._spin_from = None
        assert canvas._volume_motion(_event(inaxes=canvas.axes_at(0, 0))) is True
        assert canvas._view_angles is None
    with subtests.test(msg="axes_that_cannot_turn"):
        class _NotAnAxes3D:
            def get_zlim(self):
                return (0.0, 1.0)

        canvas = make()
        canvas._spin_from = (0.0, 0.0)
        with monkeypatch.context() as mp:
            mp.setattr(canvas, "axes_at", lambda _r, _c: _NotAnAxes3D())
            assert canvas._volume_motion(_event(inaxes=object(), x=10.0, y=10.0)) is True
        assert canvas._view_angles is None
    message_cases = [
        ("depth_drag_reports_full_depth", "c", None, "motion",
         "Full depth selected — release to create the gate."),
        ("depth_drag_reports_a_slab", "c", _Transform((0.0, 0.0), (0.0, 100.0)), "motion",
         "slab"),
        ("release_of_unreadable_depth", "not_a_column", None, "release",
         "That depth cannot be read from this angle. Choose Spin, turn the normal "
         "axis into view, then choose Draw and drag the depth again."),
    ]
    for case_id, axis, transform, which, want in message_cases:
        with subtests.test(msg=case_id):
            canvas = make()
            ax = canvas.axes_at(0, 0)
            low, high = (float(v) for v in ax.get_zlim3d())
            canvas._pending_volume_axis = axis
            canvas._depth_drag_from = (0.0, 0.0)
            said = []
            canvas.depth_requested.connect(said.append)
            real = ax.transData
            if transform is not None:
                ax.transData = transform
            try:
                handler = canvas._volume_motion if which == "motion" else canvas._volume_release
                assert handler(_event(x=0.0, y=50.0 if transform else 0.0)) is True
            finally:
                ax.transData = real
            if want == "slab":
                assert said == [f"Depth {low:.4g} to {low + (high - low) * 0.5:.4g} — "
                                "release to create the gate."]
            else:
                assert said == [want]
            if which == "release":
                assert canvas._depth_drag_from is None
    with subtests.test(msg="unreadable_depth_keeps_the_footprint"):
        canvas = make()
        said = []
        canvas.depth_requested.connect(said.append)
        cylinder = CylinderGate(name="(unnamed)", u_column="a", v_column="b",
                                axis_column="c", u_centre=1.0, v_centre=1.0,
                                u_radius=1.0, v_radius=1.0)
        canvas._pending_volume_gate, canvas._pending_volume_axis = cylinder, "c"
        canvas._depth_drag_from = (0.0, 0.0)
        with monkeypatch.context() as mp:
            mp.setattr(editor, "_project", lambda _ax, point: (point[0], point[1]))
            ax = canvas.axes_at(0, 0)
            real = ax.transData
            ax.transData = _Transform((0.0, 0.0))
            try:
                assert canvas._volume_release(_event(x=0.0, y=40.0)) is True
            finally:
                ax.transData = real
        assert "Spin" in said[0] and "Draw" in said[0]
        assert canvas._pending_volume_gate is cylinder
    with subtests.test(msg="sweep_that_described_no_volume"):
        canvas = make()
        canvas._volume_drag = ("a", 5.0, "b", 5.0)
        with monkeypatch.context() as mp:
            mp.setattr(canvas, "screen_to_volume", lambda _event: None)
            assert canvas._volume_release(_event(inaxes=None)) is True
        assert (canvas._volume_drag, canvas._pending_volume_gate) == (None, None)
    with subtests.test(msg="sweep_with_no_extent"):
        canvas = make()
        start = ("a", 5.0, "b", 5.0)
        canvas._volume_drag = start
        with monkeypatch.context() as mp:
            mp.setattr(canvas, "screen_to_volume", lambda _event: start)
            assert canvas._gate_from_volume_drag(_event(inaxes=None)) is None
        canvas._volume_drag = None
        assert canvas._gate_from_volume_drag(_event(inaxes=None)) is None
    with subtests.test(msg="preview_the_view_cannot_place"):
        canvas = make()
        canvas._volume_drag = None
        canvas._show_volume_drag(_event())
        assert canvas._ghost == []
    for case_id, x, y, read in (("plane_repeats_a_measurement", "a", "a", None),
                                ("depth_repeats_a_measurement", "a", "b", ("a", 6.0, "b", 6.0))):
        with subtests.test(msg=case_id):
            canvas = make(x=x, y=y, z="a")
            canvas._volume_drag = ("a", 4.0, y, 4.0)
            with monkeypatch.context() as mp:
                if read is not None:
                    mp.setattr(canvas, "screen_to_volume", lambda _event: read)
                canvas._show_volume_drag(_event(x=10.0, y=10.0))
                assert canvas._ghost == []
                assert canvas._gate_from_volume_drag(_event(x=10.0, y=10.0)) is None
    with subtests.test(msg="wheel"):
        canvas = make()
        before = canvas._volume_zoom
        assert canvas._volume_scroll(_event(step=1)) is True
        assert canvas._volume_zoom > before
        flat = make(y="")
        before = flat._volume_zoom
        flat._volume_scroll(_event(step=1))
        assert flat._volume_zoom == before
    with subtests.test(msg="vertex_on_a_plane_no_longer_shown"):
        canvas = make()
        ax = canvas.axes_at(0, 0)
        canvas._pending_plane = ("a", "b")
        canvas.set_mode("3D", z_column="a")
        canvas.render_now()
        assert canvas._volume_face_point(canvas.axes_at(0, 0), 5.0, 5.0) is None
        canvas._pending_plane = ()
        assert canvas._volume_face_point(ax, 5.0, 5.0) is None
    with subtests.test(msg="affine_fallback_when_the_ray_will_not_invert"):
        from mpl_toolkits.mplot3d import proj3d

        canvas = make()
        with monkeypatch.context() as mp:
            mp.setattr(proj3d, "inv_transform", lambda *_a, **_k: (_ for _ in ()).throw(
                AttributeError("this matplotlib spells its projection elsewhere")))
            first, x, second, y = canvas.screen_to_volume(_event(x=120.0, y=140.0))
        assert (first, second) == ("a", "b")
        assert np.isfinite(x) and np.isfinite(y)
    near_cases = [
        ("no_axes_under_the_click", None),
        ("projection_refuses", type("_Axes", (), {"transData": _Transform(error=RuntimeError("x"))})()),
    ]
    for case_id, inaxes in near_cases:
        with subtests.test(msg=case_id):
            canvas = make(mode="2D")
            canvas._pending = [(1.0, 2.0)]
            canvas._pending_plane = ("a", "b")
            with monkeypatch.context() as mp:
                mp.setattr(canvas, "_volume_face_point", lambda ax, u, v: (1.0, 2.0, 3.0))
                event = types.SimpleNamespace(inaxes=inaxes, x=10, y=10)
                assert canvas._near_first_volume_vertex(event) is False


def _table(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"area": rng.normal(4000.0, 600.0, n),
                         "ratio": rng.normal(5.0, 2.0, n),
                         "depth": rng.normal(-3.0, 1.0, n)})


def _box(name="kept", **bounds):
    fields = dict(x_column="area", y_column="ratio", z_column="depth",
                  x_low=3500.0, x_high=4500.0, y_low=4.0, y_high=6.0,
                  z_low=-4.0, z_high=-2.0)
    fields.update(bounds)
    return BoxGate(name=name, **fields)


def _edges_of(axes):
    out = []
    for line in axes.lines:
        xs, ys = line.get_data()
        zs = np.asarray(line.get_data_3d()[2], dtype=float)
        out.append((np.asarray(xs, float), np.asarray(ys, float), zs))
    return out


def _ringed(axes):
    clouds = [c for c in axes.collections if hasattr(c, "_offsets3d")]
    return [len(np.asarray(c._offsets3d[0])) for c in clouds[1:]]


def test_volume_gates_are_drawn_where_their_numbers_are(qtbot, subtests):
    """A box is twelve edges at its bounds; off, empty or elsewhere draws none."""
    def volume(gates, extra=None):
        canvas = GateCanvas()
        qtbot.addWidget(canvas)
        frame = _table()
        if extra:
            frame["other"] = frame["depth"] * 2.0
        canvas.set_frame(frame)
        canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
        canvas.set_mode("3D", z_column="depth")
        canvas.set_gates(gates)
        return canvas

    with subtests.test(msg="twelve_edges_at_the_gate_numbers"):
        canvas = volume(GateSet().add(_box()))
        edges = _edges_of(canvas.axes_at(0, 0))
        assert len(edges) == 12
        for axis, want in ((0, [3500.0, 4500.0]), (1, [4.0, 6.0]), (2, [-4.0, -2.0])):
            assert sorted(set(np.round(np.concatenate([e[axis] for e in edges]), 6))) == want
    with subtests.test(msg="open_sides_span_the_data"):
        canvas = volume(GateSet().add(_box(z_low=None, z_high=None)))
        zs = np.concatenate([e[2] for e in _edges_of(canvas.axes_at(0, 0))])
        depth = canvas.population()["depth"]
        assert (min(zs), max(zs)) == pytest.approx((float(depth.min()), float(depth.max())))
    with subtests.test(msg="toggled_off_then_on"):
        canvas = volume(GateSet().add(_box()))
        canvas.set_gate_enabled("kept", False)
        assert _edges_of(canvas.axes_at(0, 0)) == []
        canvas.set_gate_enabled("kept", True)
        assert len(_edges_of(canvas.axes_at(0, 0))) == 12
    with subtests.test(msg="other_third_measurement_is_ringed_not_boxed"):
        gates = GateSet().add(_box(z_column="other"))
        canvas = volume(gates, extra=True)
        axes = canvas.axes_at(0, 0)
        inside = int(gates.mask(canvas.population(), "kept").sum())
        assert _edges_of(axes) == []
        assert inside > 0 and _ringed(axes) == [inside]
    with subtests.test(msg="selects_nothing_draws_nothing"):
        canvas = volume(GateSet().add(_box(x_low=1e9, x_high=2e9)))
        assert (_edges_of(canvas.axes_at(0, 0)), _ringed(canvas.axes_at(0, 0))) == ([], [])


JACOBIAN = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
IDENTITY = tuple(map(tuple, np.eye(4)))


def _unit_box():
    return BoxGate.from_limits("b", ("a", "b", "c"), [(0.0, 1.0)] * 3)


def _ellipsoid(radius=2.0):
    return _EllipsoidGate(name="e", x_column="a", y_column="b", z_column="c",
                          x_radius=radius, y_radius=radius, z_radius=radius)


def _view(projection=IDENTITY, z="c", vertices=((-0.5, -0.5), (0.5, -0.5), (0.0, 0.5))):
    return ViewGate(name="v", x_column="a", y_column="b", z_column=z,
                    projection=projection, vertices=vertices)


def test_volume_gate_handles_move_one_face_or_the_whole_gate(screen, monkeypatch, subtests):  # noqa: F811
    """_handle_gate/_volume_handles/_handle_moved/_moved_gate on a real 3D screen."""
    canvas = screen.gates.canvas
    ax = canvas.axes_at(0, 0)

    def pull(gate, role, x=5.0, y=0.0):
        canvas._handle_drag = (gate.name, role, (0.0, 0.0), gate, np.zeros(3))
        try:
            return canvas._handle_moved(_Mouse(ax, x, y))
        finally:
            canvas._handle_drag = None

    with monkeypatch.context() as mp:
        mp.setattr(canvas, "_plot_columns", lambda: ("a", "b", "c"))
        mp.setattr(canvas, "_pixels_per_unit", lambda ax, point: JACOBIAN)
        pulls = [
            ("box_x_high", _unit_box(), "x_high", lambda g: g.x_high, 6.0),
            ("box_z_high_has_no_screen_motion", _unit_box(), "z_high", lambda g: g, _unit_box()),
            ("ellipsoid_x_low_radius", _ellipsoid(), "x_low", lambda g: g.x_radius, 3.0),
            ("view_gate_side_is_not_a_handle", _view(), "x_high", lambda g: g, _view()),
        ]
        for case_id, gate, role, read, want in pulls:
            with subtests.test(msg=case_id):
                assert read(pull(gate, role)) == (pytest.approx(want) if isinstance(want, float) else want)
        with subtests.test(msg="whole_gate_moves"):
            moved = canvas._moved_gate(_unit_box(), {"a": 1.0, "b": 2.0, "c": 3.0},
                                       np.zeros(3), np.zeros(3))
            assert (moved.x_low, moved.x_high, moved.y_high, moved.z_low) == (1.0, 2.0, 3.0, 3.0)
            centred = canvas._moved_gate(_EllipsoidGate(
                name="e", x_column="a", y_column="b", z_column="c", x_centre=0.0,
                y_centre=0.0, z_centre=0.0, x_radius=1.0, y_radius=1.0, z_radius=1.0),
                {"a": 1.0, "b": 2.0, "c": 3.0}, np.zeros(3), np.zeros(3))
            assert (centred.x_centre, centred.y_centre, centred.z_centre) == (1.0, 2.0, 3.0)
            flat = ThresholdGate(name="t", column="a", low=0.0)
            assert canvas._moved_gate(flat, {}, np.zeros(3), np.zeros(3)) is flat
            assert canvas._moved_gate(_view(), {}, np.zeros(3), np.array([1.0, 0.0, 0.0])) != _view()
            blind = _view(projection=IDENTITY[:3] + ((0.0, 0.0, 0.0, 0.0),))
            assert canvas._moved_gate(blind, {}, np.zeros(3), np.array([1.0, 0.0, 0.0])) == blind
        with subtests.test(msg="volume_handles"):
            assert canvas._volume_handles(ThresholdGate(name="t", column="a", low=0.0)) == []
            assert [role for role, _ in canvas._volume_handles(BoxGate.from_limits(
                "box", ("a", "b", "c"), [(-1, 1)] * 3))] == [
                "centre", "x_low", "x_high", "y_low", "y_high", "z_low", "z_high"]
            mp.setattr(ViewGate, "mask", lambda self, frame: (_ for _ in ()).throw(
                ValueError("no columns")))
            assert canvas._volume_handles(_view()) == []
    with subtests.test(msg="moving_a_handle_needs_drag_axes_and_projection"):
        canvas._handle_drag = None
        assert canvas._handle_moved(_Mouse(ax, 1, 1)) is None
        canvas._handle_drag = ("e", "x_high", (0.0, 0.0), _ellipsoid(1.0), np.zeros(3))
        with monkeypatch.context() as mp:
            mp.setattr(canvas, "_pixels_per_unit", lambda ax, point: None)
            assert canvas._handle_moved(_Mouse(ax, 1, 1)) is None
            mp.setattr(canvas, "axes_at", lambda r, c: None)
            assert canvas._handle_moved(_Mouse(ax, 1, 1)) is None
        canvas._handle_drag = None
    with subtests.test(msg="no_handles_means_nothing_to_grab"):
        with monkeypatch.context() as mp:
            mp.setattr(canvas, "_handle_gate", lambda: _ellipsoid(1.0))
            mp.setattr(canvas, "_volume_handles", lambda gate: [])
            assert canvas._grab_volume_handle(_Mouse(ax, 10, 10)) is False
            mp.setattr(canvas, "_volume_handles", lambda gate: [("x_high", np.zeros(3))])
            mp.setattr(editor, "_screen_points", lambda ax, points: None)
            assert canvas._grab_volume_handle(_Mouse(ax, 10, 10)) is False
            mp.setattr(editor, "_screen_points", lambda ax, points: np.array([[500.0, 500.0]]))
            assert canvas._grab_volume_handle(_Mouse(ax, 10, 10)) is False
    with subtests.test(msg="handles_need_an_enabled_closed_known_gate"):
        box, flat = _unit_box(), ThresholdGate(name="t", column="a", low=0.0)
        canvas._gates = GateSet([box, flat])
        with monkeypatch.context() as mp:
            mp.setattr(canvas, "_shows_columns", lambda gate: True)
            canvas._active = "b"
            mp.setattr(canvas, "is_gate_enabled", lambda name: False)
            assert canvas._handle_gate() is None
            mp.setattr(canvas, "is_gate_enabled", lambda name: True)
            canvas._volume_preview = replace(box, x_low=None)
            assert canvas._handle_gate() is None
            canvas._volume_preview = None
            assert canvas._handle_gate() == box
            canvas._active = "t"
            assert canvas._handle_gate() is None
    with subtests.test(msg="edit_mode_offers_every_anchor_of_a_gate_on_these_axes"):
        from PySide6.QtWidgets import QApplication

        box = BoxGate.from_limits("box", ("a", "b", "c"), [(-1, 1)] * 3)
        screen.gates.set_gates(GateSet([box]))
        screen.gates.tree.select("box")
        screen.gates._drag_buttons["edit"].click()
        for _ in range(3):
            QApplication.processEvents()
        canvas.render_now()
        ax = canvas.axes_at(0, 0)
        assert len(canvas._volume_handles(box)) == 26
        elsewhere = _view(z="d", vertices=((-1, -1), (1, -1), (1, 1), (-1, 1)))
        assert canvas._editable_boundary(elsewhere) is None
        assert canvas._volume_handles(elsewhere) == []
    with subtests.test(msg="unprojectable_point_gives_no_edit_displacement"):
        canvas._edit_drag_projection = None
        assert canvas._volume_edit_delta(ax, np.full(3, np.nan), np.asarray([10., 0.])) is None
        assert canvas._surface_moved(types.SimpleNamespace(
            inaxes=ax, x=0.0, y=0.0, xdata=0.0, ydata=0.0, button=1, step=0,
            key=None, dblclick=False)) is None
    with subtests.test(msg="a_clashing_preview_leaves_the_saved_gates"):
        canvas._gates = GateSet([_unit_box()])
        canvas._volume_preview = replace(_unit_box(), name="orphan", parent="missing")
        with monkeypatch.context() as mp:
            mp.setattr(canvas, "_draw_volume_gate", lambda *a, **k: None, raising=False)
            try:
                canvas._draw_volume_gates(ax, canvas.population(), {"fg": "#000"})
            except Exception:  # noqa: BLE001 - the saved set is the assertion
                pass
        canvas._volume_preview = None
        assert list(canvas._gates.names) == ["b"]


def test_an_editable_volume_gate_paints_a_mask_with_bounded_history(subtests):
    """VolumeMaskSession.apply_geometry: voxels painted, undo depth and byte budget."""
    from spacr.qt import mask_engine as engine
    from spacr.qt.widgets.gate_spec import EditableVolumeGate

    def box(limits, name="900"):
        return EditableVolumeGate.from_gate(BoxGate.from_limits(name, ("X", "Y", "Z"), limits))

    def empty(shape=(5, 5, 5)):
        return engine.VolumeMaskSession(np.zeros(shape), np.zeros(shape, dtype=np.uint16))

    with subtests.test(msg="between_voxel_centres_paints_nothing_keeps_geometry"):
        session = empty()
        gate = box([(0.1, 0.4), (0, 5), (0, 5)])
        assert session.apply_geometry(900, gate) == 0
        assert not session.mask.any()
        assert session.geometries == {900: gate}
        assert len(session._undo) == 1
        assert session.undo() and session.geometries == {}
    with subtests.test(msg="cancellation_changes_nothing"):
        session = empty()
        session._cancelled = lambda: True
        with pytest.raises(InterruptedError, match="cancelled"):
            session.apply_geometry(900, box([(0.1, 0.4), (0, 5), (0, 5)]))
        assert not session.mask.any() and not session.geometries and not session._undo
    with subtests.test(msg="same_geometry_twice_records_once"):
        session = empty()
        gate = box([(0, 2)] * 3)
        assert [session.apply_geometry(900, gate), session.apply_geometry(900, gate)] == [8, 0]
        assert len(session._undo) == 1 and int(session.mask.sum()) == 8 * 900
    with subtests.test(msg="byte_budget_evicts_the_oldest"):
        session = empty()
        session.apply_geometry(900, box([(0, 1)] * 3))
        one_voxel = session._history_bytes
        session.MAX_HISTORY_BYTES = one_voxel * 1.5
        session.apply_geometry(901, box([(3, 4)] * 3, name="901"))
        assert len(session._undo) == 1 and session._history_bytes == one_voxel
        assert session.undo()
        assert (session.mask[0, 0, 0], session.mask[3, 3, 3]) == (900, 0)
        assert not session.undo()
        assert session.mask[0, 0, 0] == 900
    with subtests.test(msg="undo_depth_is_twenty_five"):
        session = empty((1, 1, 30))
        for index in range(26):
            session.apply_geometry(900, box([(index, index + 1), (0, 1), (0, 1)]))
        assert len(session._undo) == 25
        assert session._history_bytes == sum(sum(v.nbytes for v in row[:3]) for row in session._undo)
        while session.undo():
            pass
        assert session.mask[0, 0, 0] == 900 and int(session.mask.sum()) == 900
