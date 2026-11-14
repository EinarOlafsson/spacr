"""The volume itself: when it is drawn, how a crowd is drawn, where a click lands.

These drive a REAL ``Axes3D`` wherever the answer is matplotlib's to give:
whether a click on a turned volume comes back as the measurement under the
cursor can only be answered by the real projection.  Stand-ins are used only
for the refusals (a crowd too extreme to bin, a matplotlib whose drag handler
cannot be wrapped), because matplotlib cannot be asked to fail.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.gate_editor import GateCanvas, _project  # noqa: E402
from spacr.qt.widgets.gate_settings import GateEditorSettings  # noqa: E402

pytestmark = pytest.mark.qt


def _table(n=300, seed=0, depth=-3.0, depth_sd=1.0):
    """Three measurements on three different scales."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"area": rng.normal(4000.0, 600.0, n),
                         "ratio": rng.normal(5.0, 2.0, n),
                         "depth": rng.normal(depth, depth_sd, n)})


@pytest.fixture
def make(qtbot):
    """A canvas over ``frame`` with area/ratio on x/y, drawn in ``mode``."""
    def build(frame=None, mode="3D", settings=None):
        canvas = GateCanvas()
        qtbot.addWidget(canvas)
        if settings is not None:
            canvas.apply_settings(settings)
        if frame is not None:
            canvas.set_frame(frame)
            canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
            canvas.set_mode(mode, z_column="depth")
        return canvas
    return build


class _Click:
    """The bits of a matplotlib mouse event the volume reads: pixels."""

    def __init__(self, x, y, inaxes=None):
        self.x, self.y = float(x), float(y)
        self.inaxes = inaxes
        self.xdata = self.ydata = None


def _pixels_of(axes, point):
    """Where a data point lands on screen, in the canvas' own pixels."""
    return axes.transData.transform(_project(axes, point))


def _limits(axes):
    return (axes.get_xlim3d(), axes.get_ylim3d(), axes.get_zlim3d())


class _Recorder:
    """Enough of an Axes3D to see what ``_draw_voxels`` would have drawn."""

    def __init__(self):
        self.calls = []

    def scatter(self, x, y, z, **kw):
        self.calls.append((np.asarray(x), kw))


class _ReadOnlyEvent:
    """A drag matplotlib will not let us rewrite."""

    def __init__(self, x, y):
        self._x, self._y = x, y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y


class _Axes:
    """A matplotlib whose drag handler can be wrapped."""

    def __init__(self):
        self._sx, self._sy = 100, 100
        self.seen = []

    def _on_move(self, event):
        self.seen.append((event.x, event.y))


class _SealedAxes:
    """A matplotlib whose drag handler cannot be replaced."""

    def __init__(self):
        self._sx, self._sy = 100, 100
        self.seen = []

    def _record(self, event):
        self.seen.append((event.x, event.y))

    @property
    def _on_move(self):
        return self._record


def test_only_a_drawable_volume_replaces_the_flat_plot(make, subtests):
    """A typo'd mode, an empty table or a 2D mode keep the flat plot."""
    for case_id, frame, mode in (("a_mode_nobody_offers", _table(), "sideways"),
                                 ("a_volume_with_no_rows", _table().iloc[0:0], "3D"),
                                 ("a_flat_plot", _table(), "2D")):
        with subtests.test(msg=case_id):
            canvas = make(frame, mode=mode)
            axes = canvas.axes_at(0, 0)
            assert not hasattr(axes, "get_zlim3d")
            if len(frame):
                assert axes.get_xlabel() == "area"
            assert canvas.box_from_view() is None


def test_a_crowd_is_drawn_as_voxel_occupancy(make, subtests):
    """Past VOXEL_THRESHOLD the cloud becomes voxels whose counts sum to the rows."""
    with subtests.test(msg="a_large_crowd_draws_fewer_marks_than_rows"):
        canvas = make(settings=GateEditorSettings(voxel_bins=8))
        canvas.set_frame(_table(canvas.VOXEL_THRESHOLD + 1))
        canvas.set_spec(replace(canvas._spec, x="area", y="ratio"))
        canvas.set_mode("3D", z_column="depth")
        axes = canvas.axes_at(0, 0)
        clouds = [c for c in axes.collections if hasattr(c, "_offsets3d")]
        xs = np.asarray(clouds[0]._offsets3d[0], dtype=float)
        assert 0 < len(xs) < 8 ** 3
    with subtests.test(msg="values_too_extreme_to_bin_draw_nothing_and_say_so"):
        canvas = make(settings=GateEditorSettings(voxel_bins=8))
        count = canvas.VOXEL_THRESHOLD + 1
        extreme = np.where(np.arange(count) % 2, 1e308, -1e308)
        axes = _Recorder()
        with np.errstate(over="ignore", invalid="ignore"):
            assert canvas._draw_voxels(axes, extreme, extreme, extreme) is False
        assert axes.calls == []
    rng = np.random.default_rng(1)
    canvas = make(settings=GateEditorSettings(voxel_bins=6))
    count = canvas.VOXEL_THRESHOLD + 1
    outlier = rng.normal(0.0, 1.0, count)
    outlier[0] = 1e6
    clouds = {
        "identical": np.full(count, 7.0),
        "one_outlier": outlier,
        "two_clumps": np.where(np.arange(count) % 2, rng.normal(-50, 1, count),
                               rng.normal(50, 1, count)),
    }
    for case_id, values in clouds.items():
        with subtests.test(msg=f"every_object_lands_in_a_voxel_{case_id}"):
            axes = _Recorder()
            assert canvas._draw_voxels(axes, values, values, values) is True
            assert np.asarray(axes.calls[0][1]["c"], dtype=float).sum() == count


def test_the_spin_speed_wrap_never_costs_the_rotation(make, subtests):
    """A drag the wrapper cannot rescale still reaches matplotlib, unscaled."""
    for case_id, axes_type, event, speed in (
            ("a_read_only_event", _Axes, _ReadOnlyEvent(140, 100), 3.0),
            ("a_handler_that_cannot_be_replaced", _SealedAxes, _ReadOnlyEvent(110, 100), 2.0)):
        with subtests.test(msg=case_id):
            canvas = make(settings=GateEditorSettings(spin_speed=speed))
            axes = axes_type()
            canvas._apply_spin_speed(axes)
            axes._on_move(event)
            assert axes.seen == [(event.x, event.y)]
            if axes_type is _SealedAxes:
                assert not getattr(axes._on_move, "_spacr_wrapped", False)


def test_a_click_on_the_volume_lands_on_the_measurement_under_it(make, subtests):
    """``screen_to_volume`` inverts the real projection to 1e-4 of the axis."""
    canvas = make(_table())
    axes = canvas.axes_at(0, 0)
    index = {"area": 0, "ratio": 1, "depth": 2}
    for elev, azim in ((0.0, 0.0), (30.0, 40.0), (17.0, 213.0), (-25.0, 97.0)):
        for axis in ("x", "y", "z"):
            with subtests.test(msg=f"turned_{elev}_{azim}_on_{axis}"):
                axes.view_init(elev=elev, azim=azim)
                canvas._canvas.draw()
                canvas.set_anchor_axis(axis)
                first, second, normal = canvas.anchor_plane()
                limits = _limits(axes)
                point = [0.0, 0.0, 0.0]
                for column, fraction in ((first, 0.25), (second, 0.4)):
                    low, high = limits[index[column]]
                    point[index[column]] = low + fraction * (high - low)
                point[index[normal]] = limits[index[normal]][0]
                got = canvas.screen_to_volume(_Click(*_pixels_of(axes, point)))
                assert (got[0], got[2]) == (first, second)
                for name, value in ((got[0], got[1]), (got[2], got[3])):
                    low, high = limits[index[name]]
                    assert abs(value - point[index[name]]) < (high - low) * 1e-4
    with subtests.test(msg="a_click_before_the_first_paint"):
        canvas = make(_table())
        axes = canvas.axes_at(0, 0)
        axes.M = None
        canvas.set_anchor_axis("z")
        limits = _limits(axes)
        point = [limits[0][0] + 0.3 * (limits[0][1] - limits[0][0]),
                 limits[1][0] + 0.6 * (limits[1][1] - limits[1][0]),
                 limits[2][0]]
        got = canvas.screen_to_volume(_Click(*_pixels_of(axes, point)))
        assert (got[0], got[2]) == ("area", "ratio")
        assert (got[1], got[3]) == pytest.approx((point[0], point[1]), rel=1e-6)
    with subtests.test(msg="the_first_polygon_vertex_stays_under_the_cursor"):
        canvas = make(_table(seed=3, depth=-800.0, depth_sd=40.0))
        canvas.set_anchor_axis("x")
        canvas.set_drag_mode("draw")
        canvas.set_volume_shape("polygon")
        axes = canvas.axes_at(0, 0)
        columns = ("area", "ratio", "depth")

        def face_point(u, v):
            first, second, normal = canvas.anchor_plane()
            point = [0.0, 0.0, 0.0]
            point[columns.index(first)] = float(u)
            point[columns.index(second)] = float(v)
            point[columns.index(normal)] = float(_limits(axes)[columns.index(normal)][0])
            return point

        shape = ((3.0, -820.0), (7.0, -820.0), (5.0, -780.0))
        clicked = None
        for u, v in shape:
            pixels = _pixels_of(axes, face_point(u, v))
            canvas._on_press(_Click(pixels[0], pixels[1], axes))
            clicked = pixels if clicked is None else clicked
        now = _pixels_of(axes, face_point(*shape[0]))
        drift = float(np.hypot(now[0] - clicked[0], now[1] - clicked[1]))
        assert drift <= canvas.CLOSE_RADIUS_PX
