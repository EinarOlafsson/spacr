"""The 3D volume turns freely and stays where it is let go.

The maintainer, 2026-09-21: "in gate editor whenever i spin the graph always
snapps to a n angle where 1 on the exes (in this cans the y axis) is flat on
the bottom and the graph en only spins on one axis and always snaps to a
position."

Two causes: ``snap_to_axis`` defaulted to on, so every release squared the
camera up to the nearest face (elevation 0, azimuth a multiple of 90 -- one
measurement lying flat along the bottom), and the spin was locked to Z by
default, so a drag could only change the azimuth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PySide6.QtWidgets import QApplication

from spacr.qt.linked_selection import LinkedSelection
from spacr.qt.widgets.gate_settings import GateEditorSettings
from spacr.qt.widgets.volume_view import (
    angles_from_axes, rotate_about_world, trackball, view_axes,
)


@pytest.fixture
def screen(qtbot):
    from spacr.qt.screens.gate_editor import GateEditorScreen

    widget = GateEditorScreen(link=LinkedSelection(), threaded=False)
    qtbot.addWidget(widget)
    rng = np.random.default_rng(0)
    widget.set_frame(pd.DataFrame({"a": rng.normal(0, 1, 300),
                                   "b": rng.normal(0, 1, 300),
                                   "c": rng.normal(0, 1, 300)}))
    widget._x.setCurrentText("a")
    widget._y.setCurrentText("b")
    widget.gates.mode_requested.emit("3D")
    widget._z.setCurrentText("c")
    return widget


class _Mouse:
    def __init__(self, ax, x, y, button=1):
        self.inaxes, self.x, self.y = ax, float(x), float(y)
        self.xdata = self.ydata = 0.0
        self.button = button
        self.step = 0


def _drag(canvas, start, end, steps=6):
    ax = canvas.axes_at(0, 0)
    canvas._on_press(_Mouse(ax, *start))
    for i in range(1, steps + 1):
        t = i / steps
        canvas._on_motion(_Mouse(ax, start[0] + (end[0] - start[0]) * t,
                                 start[1] + (end[1] - start[1]) * t))
    release = _Mouse(ax, *end)
    canvas._on_release(release)
    canvas._on_button_release(release)


def _angles(ax):
    return (float(ax.elev), float(ax.azim), float(ax.roll))


def test_snapping_is_off_by_default():
    assert GateEditorSettings().snap_to_axis is False


def test_the_default_spin_is_free(screen):
    assert screen.gates.canvas._spin_axis == ""
    assert screen.gates._spin_buttons[""].isChecked()


def test_a_released_spin_stays_where_it_was_let_go(screen, qtbot):
    canvas = screen.gates.canvas
    _drag(canvas, (100, 100), (157, 131))
    ax = canvas.axes_at(0, 0)
    released = _angles(ax)
    for _ in range(20):
        QApplication.processEvents()
        qtbot.wait(5)
    ax = canvas.axes_at(0, 0)
    assert _angles(ax) == pytest.approx(released)
    assert released[0] not in (0.0, 90.0, -90.0)
    assert released[1] % 90.0 != pytest.approx(0.0)


def test_the_angle_survives_a_redraw(screen):
    canvas = screen.gates.canvas
    _drag(canvas, (100, 100), (140, 170))
    released = _angles(canvas.axes_at(0, 0))
    canvas.render_now()
    assert _angles(canvas.axes_at(0, 0)) == pytest.approx(released)


def test_one_drag_turns_about_both_screen_axes(screen):
    canvas = screen.gates.canvas
    ax = canvas.axes_at(0, 0)
    before = _angles(ax)
    _drag(canvas, (100, 100), (160, 160))
    after = _angles(canvas.axes_at(0, 0))
    assert after[0] != pytest.approx(before[0])
    assert after[1] != pytest.approx(before[1])


def test_a_vertical_drag_can_go_past_the_pole(screen):
    """Every orientation is reachable: the old clamp stopped at 90."""
    canvas = screen.gates.canvas
    ax = canvas.axes_at(0, 0)
    u0, v0, w0 = view_axes(*_angles(ax))
    _drag(canvas, (100, 100), (100, 460), steps=30)
    u1, v1, w1 = view_axes(*_angles(canvas.axes_at(0, 0)))
    # A turn of 180 degrees about the screen horizontal flips the view.
    assert float(w0 @ w1) == pytest.approx(-1.0, abs=1e-6)


def test_reset_view_returns_to_the_default(screen):
    canvas = screen.gates.canvas
    default = _angles(canvas.axes_at(0, 0))
    _drag(canvas, (100, 100), (170, 150))
    screen.gates.reset_view()
    assert canvas._view_angles is None
    assert _angles(canvas.axes_at(0, 0)) == pytest.approx(default)


def test_the_snap_setting_still_snaps_when_asked_for(screen):
    canvas = screen.gates.canvas
    canvas.apply_settings(GateEditorSettings(snap_to_axis=True))
    _drag(canvas, (100, 100), (157, 131))
    elev, azim, roll = _angles(canvas.axes_at(0, 0))
    assert elev in (0.0, 90.0, -90.0)
    assert azim % 90.0 == pytest.approx(0.0)


def test_a_right_button_drag_spins_even_while_drawing(screen):
    canvas = screen.gates.canvas
    canvas.set_drag_mode("draw")
    ax = canvas.axes_at(0, 0)
    before = _angles(ax)
    canvas._on_press(_Mouse(ax, 100, 100, button=3))
    canvas._on_motion(_Mouse(ax, 150, 130, button=3))
    canvas._on_release(_Mouse(ax, 150, 130, button=3))
    assert _angles(canvas.axes_at(0, 0)) != pytest.approx(before)


# ---------------------------------------------------------------------------
# The camera arithmetic against matplotlib's own
# ---------------------------------------------------------------------------

def test_view_axes_match_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    rng = np.random.default_rng(1)
    for _ in range(100):
        e, a, r = (rng.uniform(-90, 90), rng.uniform(-180, 180),
                   rng.uniform(-180, 180))
        ax.view_init(e, a, r)
        ax.get_proj()
        u, v, w = view_axes(e, a, r)
        assert np.allclose(u, ax._view_u)
        assert np.allclose(v, ax._view_v)
        assert np.allclose(w, ax._view_w)
        u2, _v2, w2 = view_axes(*angles_from_axes(u, w))
        assert np.allclose(u, u2) and np.allclose(w, w2)
    plt.close(fig)


def test_straight_down_is_representable():
    elev, azim, roll = angles_from_axes([0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    u, _v, w = view_axes(elev, azim, roll)
    assert np.allclose(u, [0, 1, 0], atol=1e-5)
    assert np.allclose(w, [0, 0, 1], atol=1e-5)


def test_a_z_lock_changes_only_the_azimuth():
    elev, azim, roll = rotate_about_world(20.0, 30.0, 0.0, "z", 15.0)
    assert (elev, roll) == pytest.approx((20.0, 0.0), abs=1e-9)
    assert azim == pytest.approx(15.0)


def test_trackball_turns_compose():
    once = trackball(10.0, 20.0, 0.0, 30.0, 0.0)
    twice = trackball(*trackball(10.0, 20.0, 0.0, 15.0, 0.0), 15.0, 0.0)
    assert np.allclose(view_axes(*once)[2], view_axes(*twice)[2])
