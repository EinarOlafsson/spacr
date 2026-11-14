"""What the mouse does inside the 3D volume: spin, sweep, depth and wheel.

Instruction 52, from the complaint that opened it:

    "i cant zoom in or spin on any of the axees. if i press pollygon and tried
     to draw a gate, then i could all of a suded spinn the graph"

Those are four gestures sharing one mouse -- turning the volume, sweeping out
a footprint, dragging the slab that gives it depth, and the wheel.  Each test
below walks one gesture's case table with pytest's ``subtests`` fixture; every
case builds its own :class:`GateCanvas` and its own stand-in Axes3D (a real
one needs a renderer and a live projection matrix before it will answer).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

import spacr.qt.widgets.gate_editor as editor  # noqa: E402
from spacr.qt.widgets.gate_editor import GateCanvas  # noqa: E402
from spacr.qt.widgets.gate_spec import BoxGate  # noqa: E402

pytestmark = pytest.mark.qt


class _Identity:
    """A transData that leaves projected coordinates where they are."""

    @staticmethod
    def transform(point):
        return (float(point[0]), float(point[1]))


class _Line:
    """One of the dashed outlines the sweep leaves behind."""

    def __init__(self, xs, ys, zs, style):
        self.xs, self.ys, self.zs = tuple(xs), tuple(ys), tuple(zs)
        self.style = style
        self.removed = False

    def remove(self):
        self.removed = True


class _Volume:
    """A stand-in for matplotlib's Axes3D: limits, camera and ``plot``."""

    def __init__(self, elev=15.0, azim=40.0):
        self.elev, self.azim = elev, azim
        self.limits = {"x": (0.0, 10.0), "y": (0.0, 20.0), "z": (0.0, 30.0)}
        self.drawn: list = []
        self.transData = _Identity()

    def get_zlim(self):
        return self.limits["z"]

    def get_xlim3d(self):
        return self.limits["x"]

    def get_ylim3d(self):
        return self.limits["y"]

    def get_zlim3d(self):
        return self.limits["z"]

    def set_xlim3d(self, low, high):
        self.limits["x"] = (float(low), float(high))

    def set_ylim3d(self, low, high):
        self.limits["y"] = (float(low), float(high))

    def set_zlim3d(self, low, high):
        self.limits["z"] = (float(low), float(high))

    def view_init(self, elev=None, azim=None):
        self.elev, self.azim = float(elev), float(azim)

    def plot(self, xs, ys, zs, **kwargs):
        line = _Line(xs, ys, zs, kwargs.get("linestyle"))
        self.drawn.append(line)
        return [line]


class _NoCamera:
    """A depth-reporting axes that cannot be turned."""

    def get_zlim(self):
        return (0.0, 30.0)


class _Event:
    """A matplotlib mouse event, reduced to what the handlers read."""

    def __init__(self, x=0.0, y=0.0, inaxes=True, **extra):
        self.x = float(x)
        self.y = float(y)
        self.inaxes = object() if inaxes else None
        for key, value in extra.items():
            setattr(self, key, value)


FRAME = pd.DataFrame({"a": [0.0, 10.0], "b": [0.0, 20.0], "c": [0.0, 30.0]})


@pytest.fixture
def make(qtbot):
    """Build a 3D canvas over FRAME (a/b/c on x/y/z) and its stand-in volume."""
    def build(with_volume=True):
        canvas = GateCanvas()
        qtbot.addWidget(canvas)
        canvas._spec = replace(canvas._spec, x="a", y="b")
        canvas._z_column = "c"
        canvas._mode = "3D"
        canvas._frame = FRAME.copy()
        volume = _Volume()
        if with_volume:
            canvas.axes_at = lambda *_a, **_k: volume
        said: list = []
        canvas.depth_requested.connect(said.append)
        return canvas, volume, said
    return build


def _sees(canvas, *corners):
    """Make the mouse read as the given points on the anchor plane."""
    points = iter(corners)
    canvas.screen_to_volume = lambda _event: next(points)


def _straight_on(mp):
    """A camera looking straight at the A axis: depth C runs up the screen."""
    mp.setattr(editor, "_project", lambda _ax, point: (point[0], point[2]))


def _edge_on(mp):
    """A camera looking down the C axis: the whole axis is one screen point."""
    mp.setattr(editor, "_project", lambda _ax, point: (point[0], point[1]))


def test_spinning_the_volume_follows_the_drag_and_its_lock(make, subtests):
    """A drag turns the camera by 0.5 degree per pixel about the locked axis."""
    with subtests.test(msg="a_press_outside_the_plot_is_swallowed_and_starts_nothing"):
        canvas, _volume, _ = make()
        canvas.set_drag_mode("draw")
        _sees(canvas, ("a", 1.0, "b", 2.0))
        assert canvas._volume_press(_Event(5.0, 6.0, inaxes=False)) is True
        assert (canvas._volume_drag, canvas._spin_from) == (None, None)
    with subtests.test(msg="locked_to_z_a_drag_changes_azimuth_only"):
        canvas, volume, _ = make()
        canvas.set_spin_axis("z")
        canvas._spin_from = (10.0, 20.0)
        assert canvas._volume_motion(_Event(30.0, 90.0)) is True
        assert (volume.elev, volume.azim) == pytest.approx((15.0, 30.0))
        assert canvas._view_angles[:2] == pytest.approx((15.0, 30.0))
        assert canvas._spin_from == (30.0, 90.0)
    for case_id, axis in (("unlocked_one_drag_turns_both_angles", ""),
                          ("an_unknown_axis_falls_back_to_free", "diagonal")):
        with subtests.test(msg=case_id):
            canvas, volume, _ = make()
            canvas.set_spin_axis(axis)
            assert canvas._spin_axis == ""
            canvas._spin_from = (10.0, 20.0)
            canvas._volume_motion(_Event(30.0, 60.0))
            # elev 15 - 40 * 0.5, azim 40 - 20 * 0.5
            assert (volume.elev, volume.azim) == pytest.approx((-5.0, 30.0))
    with subtests.test(msg="each_step_measures_from_the_last_one"):
        canvas, volume, _ = make()
        canvas.set_spin_axis("z")
        canvas._spin_from = (0.0, 0.0)
        canvas._volume_motion(_Event(20.0, 0.0))
        assert canvas._spin_from == (20.0, 0.0)
        canvas._volume_motion(_Event(40.0, 0.0))
        assert volume.azim == pytest.approx(20.0)      # 40 - 10 - 10
    with subtests.test(msg="no_button_down_turns_nothing"):
        canvas, volume, _ = make()
        assert canvas._volume_motion(_Event(90.0, 90.0)) is True
        assert (volume.elev, volume.azim) == (15.0, 40.0)
    with subtests.test(msg="a_spin_wandering_off_the_plot_holds_its_angle"):
        canvas, volume, _ = make()
        canvas.set_spin_axis("z")
        canvas._spin_from = (10.0, 20.0)
        assert canvas._volume_motion(_Event(90.0, 20.0, inaxes=False)) is True
        assert (volume.elev, volume.azim) == (15.0, 40.0)
    with subtests.test(msg="an_axes_that_cannot_turn_refuses_without_raising"):
        canvas, _volume, _ = make()
        canvas.axes_at = lambda *_a, **_k: _NoCamera()
        canvas._spin_from = (10.0, 20.0)
        assert canvas._volume_motion(_Event(90.0, 90.0)) is True
        assert canvas._spin_from == (10.0, 20.0)
    with subtests.test(msg="letting_go_ends_the_spin"):
        canvas, _volume, _ = make()
        canvas._spin_from = (10.0, 20.0)
        assert canvas._volume_release(_Event(30.0, 40.0)) is True
        assert canvas._spin_from is None
    for handler in ("_volume_motion", "_volume_release"):
        with subtests.test(msg=f"{handler}_on_a_flat_plot_is_left_to_2d"):
            canvas, _volume, _ = make()
            canvas._mode = "2D"
            assert getattr(canvas, handler)(_Event(90.0, 90.0)) is False


def test_sweeping_a_footprint_previews_it_through_the_depth(make, subtests):
    """The footprint is drawn at both depth ends on the picked plane; a plane needs three measurements."""
    with subtests.test(msg="shown_at_both_ends_of_the_depth"):
        canvas, volume, _ = make()
        canvas.set_drag_mode("draw")
        canvas._volume_drag = ("a", 1.0, "b", 2.0)
        _sees(canvas, ("a", 3.0, "b", 6.0))
        assert canvas._volume_motion(_Event(40.0, 50.0)) is True
        near, far = volume.drawn
        assert (set(near.zs), set(far.zs)) == ({0.0}, {30.0})
        assert near.xs == (1.0, 3.0, 3.0, 1.0, 1.0)
        assert near.ys == (2.0, 2.0, 6.0, 6.0, 2.0)
        assert near.style == "--"
    with subtests.test(msg="each_motion_replaces_the_outline"):
        canvas, volume, _ = make()
        canvas.set_drag_mode("draw")
        canvas._volume_drag = ("a", 1.0, "b", 2.0)
        _sees(canvas, ("a", 3.0, "b", 6.0), ("a", 5.0, "b", 9.0))
        canvas._volume_motion(_Event(40.0, 50.0))
        first_pair = list(canvas._ghost)
        canvas._volume_motion(_Event(60.0, 70.0))
        assert len(canvas._ghost) == 2
        assert [line.removed for line in first_pair] == [True, True]
        assert canvas._ghost[0].xs == (1.0, 5.0, 5.0, 1.0, 1.0)
    with subtests.test(msg="drawn_on_the_plane_the_user_picked"):
        canvas, volume, _ = make()
        canvas.set_anchor_axis("x")
        canvas.set_drag_mode("draw")
        canvas._volume_drag = ("b", 4.0, "c", 5.0)
        _sees(canvas, ("b", 8.0, "c", 15.0))
        canvas._volume_motion(_Event(40.0, 50.0))
        near, far = volume.drawn
        assert (set(near.xs), set(far.xs)) == ({0.0}, {10.0})
        assert near.ys == (4.0, 8.0, 8.0, 4.0, 4.0)
        assert near.zs == (5.0, 5.0, 15.0, 15.0, 5.0)
    with subtests.test(msg="an_unreadable_sweep_shows_nothing"):
        canvas, volume, _ = make()
        canvas._volume_drag = ("a", 1.0, "b", 2.0)
        canvas.screen_to_volume = lambda _event: None
        assert canvas._volume_motion(_Event(40.0, 50.0)) is True
        assert (volume.drawn, canvas._ghost) == ([], [])
    with subtests.test(msg="a_sweep_that_makes_nothing_clears_and_redraws"):
        canvas, volume, _ = make()
        drawn: list = []
        canvas.gate_drawn.connect(drawn.append)
        redrawn: list = []
        canvas.render_now = lambda: redrawn.append(True)
        canvas.set_drag_mode("draw")
        canvas._volume_drag = ("a", 1.0, "b", 2.0)
        _sees(canvas, ("a", 3.0, "b", 6.0), ("a", 1.0, "b", 2.0))
        canvas._volume_motion(_Event(40.0, 50.0))
        assert canvas._volume_release(_Event(40.0, 50.0)) is True
        assert (canvas._volume_drag, canvas._ghost, drawn, redrawn) == (
            None, [], [], [True])
    with subtests.test(msg="the_same_measurement_on_two_axes_refuses_the_sweep"):
        canvas, volume, _ = make()
        canvas._z_column = "a"
        canvas._volume_drag = ("a", 1.0, "b", 2.0)
        _sees(canvas, ("a", 3.0, "b", 6.0), ("a", 3.0, "b", 6.0))
        assert canvas._volume_motion(_Event(40.0, 50.0)) is True
        assert volume.drawn == []
        assert canvas._gate_from_volume_drag(_Event(40.0, 50.0)) is None
    for case_id, z_column, expected in (("duplicate_axis_has_no_plane", "a", None),
                                        ("three_distinct_arm_their_plane", "c",
                                         ("a", "b", "c"))):
        with subtests.test(msg=case_id):
            canvas, _volume, _ = make(with_volume=False)
            canvas.set_mode("3D", z_column=z_column)
            canvas._spec = replace(canvas._spec, x="a", y="b")
            canvas.set_anchor_axis("z")
            assert canvas.anchor_plane() == expected


def test_a_depth_drag_reads_as_a_fraction_of_the_visible_range(make, monkeypatch, subtests):
    """``_depth_bounds_from_drag``: a slab, full depth (None, None) or refusal."""
    cases = [
        # case_id, camera, axis, limits_z, end_event, expected
        ("half_the_axis_is_half_its_range", _straight_on, "c", None, (0.0, 15.0), (0.0, 15.0)),
        ("dragging_backwards_asks_the_same_slab", _straight_on, "c", None, (0.0, -15.0), (0.0, 15.0)),
        ("a_wobble_is_a_click_for_full_depth", _straight_on, "c", None, (2.0, 1.0), (None, None)),
        ("the_whole_way_down_is_full_depth", _straight_on, "c", None, (0.0, 29.5), (None, None)),
        ("an_axis_pointing_at_the_viewer_cannot_be_read", _edge_on, "c", None, (0.0, 40.0), None),
        ("no_visible_range_has_no_depth", _straight_on, "c", (7.0, 7.0), (0.0, 40.0), None),
        ("a_measurement_no_longer_shown_is_refused", _straight_on, "gone", None, (0.0, 15.0), None),
    ]
    for case_id, camera, axis, limits_z, end, expected in cases:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                camera(mp)
                canvas, volume, _ = make()
                if limits_z is not None:
                    volume.limits["z"] = limits_z
                canvas._pending_volume_axis = axis
                got = canvas._depth_bounds_from_drag((0.0, 0.0), _Event(*end))
            if expected is None or expected == (None, None):
                assert got == expected
            else:
                assert got == pytest.approx(expected)
    with subtests.test(msg="a_flat_plot_has_no_depth"):
        canvas, _volume, _ = make()
        canvas.axes_at = lambda *_a, **_k: object()
        canvas._pending_volume_axis = "c"
        assert canvas._depth_bounds_from_drag((0.0, 0.0), _Event(0.0, 15.0)) is None
    with subtests.test(msg="a_failing_projection_costs_only_the_depth"):
        def boom(_ax, _point):
            raise RuntimeError("this matplotlib projects differently")
        with monkeypatch.context() as mp:
            mp.setattr(editor, "_project", boom)
            canvas, _volume, _ = make()
            canvas._pending_volume_axis = "c"
            assert canvas._depth_bounds_from_drag((0.0, 0.0), _Event(0.0, 15.0)) is None


def test_the_depth_gesture_names_its_slab_and_bounds_the_gate(make, monkeypatch, subtests):
    """While dragging the depth the canvas says what it will make; release makes it."""
    readouts = [
        ("a_slab_is_named_before_commit", _straight_on, (0.0, 15.0),
         ["Depth 0 to 15 — release to create the gate."]),
        ("holding_still_is_full_depth", _straight_on, (1.0, 1.0),
         ["Full depth selected — release to create the gate."]),
        ("an_unreadable_depth_says_nothing", _edge_on, (0.0, 40.0), []),
    ]
    for case_id, camera, end, expected in readouts:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                camera(mp)
                canvas, _volume, said = make()
                canvas._depth_drag_from = (0.0, 0.0)
                canvas._pending_volume_axis = "c"
                assert canvas._volume_motion(_Event(*end)) is True
            assert said == expected
    with subtests.test(msg="sweep_then_depth_makes_one_bounded_box"):
        with monkeypatch.context() as mp:
            _straight_on(mp)
            canvas, _volume, _ = make()
            drawn: list = []
            canvas.gate_drawn.connect(drawn.append)
            canvas.set_drag_mode("draw")
            canvas.set_volume_shape("box")
            _sees(canvas, ("a", 1.0, "b", 2.0), ("a", 5.0, "b", 8.0))
            canvas._volume_press(_Event(0.0, 0.0))
            canvas._volume_release(_Event(40.0, 50.0))
            assert isinstance(canvas._pending_volume_gate, BoxGate)
            assert drawn == []
            canvas._volume_press(_Event(0.0, 0.0))
            canvas._volume_release(_Event(0.0, 15.0))
        assert len(drawn) == 1
        assert (drawn[0].z_low, drawn[0].z_high) == pytest.approx((0.0, 15.0))
        assert (canvas._pending_volume_gate, canvas._pending_volume_axis) == (None, "")


def test_the_wheel_zooms_the_volume_about_its_data(make, monkeypatch, subtests):
    """``_volume_scroll`` scales the zoom by 1.25 per notch inside [0.05, 50]."""
    with subtests.test(msg="one_notch_in_shrinks_every_axis_about_its_centre"):
        canvas, volume, _ = make()
        canvas._apply_volume_zoom(volume)
        before = dict(volume.limits)
        assert canvas._volume_scroll(_Event(0.0, 0.0, step=1)) is True
        assert canvas._volume_zoom == pytest.approx(1.25)
        for axis, centre in (("x", 5.0), ("y", 10.0), ("z", 15.0)):
            low, high = volume.limits[axis]
            assert high - low == pytest.approx((before[axis][1] - before[axis][0]) / 1.25)
            assert (low + high) / 2 == pytest.approx(centre)
    with subtests.test(msg="a_wheel_with_no_step_reads_the_button"):
        canvas, volume, _ = make()
        canvas._apply_volume_zoom(volume)
        resting = volume.limits["x"][1] - volume.limits["x"][0]
        assert canvas._volume_scroll(_Event(0.0, 0.0, step=0, button="down")) is True
        assert canvas._volume_zoom == pytest.approx(0.8)
        assert volume.limits["x"][1] - volume.limits["x"][0] == pytest.approx(resting / 0.8)
    for case_id, start, step, expected in (("clamped_at_fifty", 49.0, 1, 50.0),
                                           ("clamped_at_a_twentieth", 0.06, -1, 0.05)):
        with subtests.test(msg=case_id):
            canvas, _volume, _ = make()
            canvas._volume_zoom = start
            canvas._volume_scroll(_Event(0.0, 0.0, step=step))
            assert canvas._volume_zoom == pytest.approx(expected)
    with subtests.test(msg="a_flat_plot_is_left_to_the_2d_zoom"):
        canvas, _volume, _ = make()
        canvas._mode = "2D"
        assert canvas._volume_scroll(_Event(0.0, 0.0, step=1)) is False
    with subtests.test(msg="no_axes_left_is_swallowed_not_raised"):
        with monkeypatch.context() as mp:
            canvas, _volume, _ = make()
            mp.setattr(canvas, "_in_volume", lambda: True)
            canvas.axes_at = lambda *_a, **_k: None
            assert canvas._volume_scroll(_Event(0.0, 0.0, step=1)) is True
            assert canvas._volume_zoom == pytest.approx(1.0)


def test_the_zoomed_limits_come_from_the_shown_measurements(make, subtests):
    """``_apply_volume_zoom``: mean +/- 3 SD / zoom, skipping what cannot be read."""
    cases = [
        # case_id, frame, z_column, zoom, expected limits (None = untouched)
        ("zoom_two_halves_the_half_width", None, "c", 2.0,
         {"x": (-2.5, 12.5)}),
        ("a_constant_measurement_gets_room", pd.DataFrame(
            {"a": [7.0, 7.0], "b": [0.0, 20.0], "c": [0.0, 30.0]}), "c", 1.0,
         {"x": (6.0, 8.0)}),
        ("a_measurement_not_in_the_table_is_skipped", None, "not_measured", 1.0,
         {"z": (0.0, 30.0), "x": (-10.0, 20.0)}),
        ("an_all_blank_measurement_is_skipped", pd.DataFrame(
            {"a": [0.0, 10.0], "b": [0.0, 20.0], "c": [np.nan, np.nan]}), "c", 1.0,
         {"z": (0.0, 30.0)}),
        ("nothing_loaded_leaves_the_axes_alone", "none", "c", 1.0,
         {"x": (0.0, 10.0), "y": (0.0, 20.0), "z": (0.0, 30.0)}),
    ]
    for case_id, frame, z_column, zoom, expected in cases:
        with subtests.test(msg=case_id):
            canvas, volume, _ = make()
            if isinstance(frame, pd.DataFrame):
                canvas._frame = frame
            elif frame == "none":
                canvas._frame = None
            canvas._z_column = z_column
            canvas._volume_zoom = zoom
            canvas._apply_volume_zoom(volume)
            for axis, limits in expected.items():
                assert volume.limits[axis] == pytest.approx(limits)
