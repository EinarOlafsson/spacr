"""The 3D settings turn something. Instruction 52.

``voxel_bins``, ``snap_to_axis`` and ``spin_speed`` were declared, given a
control, saved, reloaded -- and read by nothing. A control that turns nothing
is a promise the application does not keep, which is what instruction 77's
sweep was about, and 52 set the bar itself: ``GateEditorSettings`` should end
this instruction with zero fields nothing reads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacr.qt.widgets.gate_editor import GateCanvas
from spacr.qt.widgets.gate_settings import GateEditorSettings


@pytest.fixture
def canvas(qtbot):
    widget = GateCanvas()
    qtbot.addWidget(widget)
    return widget


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n),
                         "z": rng.normal(0, 1, n)})


class _Release:
    """The bits of a matplotlib release event this handler reads."""
    x = y = 0
    inaxes = None


# ---------------------------------------------------------------------------
# snap_to_axis
# ---------------------------------------------------------------------------

def test_a_spin_that_ends_squares_the_volume_up(canvas, monkeypatch):
    canvas.apply_settings(GateEditorSettings(snap_to_axis=True))
    canvas._mode = "3D"
    canvas._view_angles = (23.0, 47.0)
    called = []
    monkeypatch.setattr(canvas, "snap_to_nearest_axis",
                        lambda: called.append(True) or (0.0, 0.0))
    canvas._on_button_release(_Release())
    assert called


def test_the_setting_off_leaves_the_view_where_the_user_left_it(canvas,
                                                                monkeypatch):
    canvas.apply_settings(GateEditorSettings(snap_to_axis=False))
    canvas._mode = "3D"
    canvas._view_angles = (23.0, 47.0)
    called = []
    monkeypatch.setattr(canvas, "snap_to_nearest_axis",
                        lambda: called.append(True) or (0.0, 0.0))
    canvas._on_button_release(_Release())
    assert not called


def test_a_volume_nobody_turned_is_not_snapped(canvas, monkeypatch):
    """Snapping a view the user set deliberately would move it for nothing."""
    canvas.apply_settings(GateEditorSettings(snap_to_axis=True))
    canvas._mode = "3D"
    canvas._view_angles = None
    called = []
    monkeypatch.setattr(canvas, "snap_to_nearest_axis",
                        lambda: called.append(True) or (0.0, 0.0))
    canvas._on_button_release(_Release())
    assert not called


def test_nothing_snaps_in_2d(canvas, monkeypatch):
    canvas.apply_settings(GateEditorSettings(snap_to_axis=True))
    canvas._mode = "2D"
    canvas._view_angles = (23.0, 47.0)
    called = []
    monkeypatch.setattr(canvas, "snap_to_nearest_axis",
                        lambda: called.append(True) or (0.0, 0.0))
    canvas._on_button_release(_Release())
    assert not called


def test_a_snap_that_fails_does_not_take_the_release_with_it(canvas,
                                                             monkeypatch):
    canvas.apply_settings(GateEditorSettings(snap_to_axis=True))
    canvas._mode = "3D"
    canvas._view_angles = (23.0, 47.0)

    def boom():
        raise RuntimeError("no axes")

    monkeypatch.setattr(canvas, "snap_to_nearest_axis", boom)
    canvas._on_button_release(_Release())      # the assertion is no raise


@pytest.mark.parametrize("elev,azim,want_elev,want_azim", [
    (23.0, 47.0, 0.0, 90.0),   # 47 is nearer 90 than 0
    (70.0, 100.0, 90.0, 90.0),
    (-80.0, 200.0, -90.0, 180.0),
])
def test_snapping_lands_on_a_face(canvas, elev, azim, want_elev, want_azim):
    """The point of snapping: a 3D gate is finally judged from a view where
    one measurement is flat.

    The arithmetic is what is under test, so the axes are a stand-in -- a
    real Axes3D would make this a test of matplotlib's projection.
    """
    class _Axes3D:
        def __init__(self):
            self.elev, self.azim = elev, azim

        def view_init(self, elev, azim):
            self.elev, self.azim = elev, azim

    axes = _Axes3D()
    canvas.axes_at = lambda *a, **k: axes
    canvas._canvas.draw_idle = lambda: None
    assert canvas.snap_to_nearest_axis() == (want_elev, want_azim)
    assert (axes.elev, axes.azim) == (want_elev, want_azim)


# ---------------------------------------------------------------------------
# spin_speed
# ---------------------------------------------------------------------------

class _Axes:
    """Enough of an Axes3D to see whether the drag was scaled."""

    def __init__(self):
        self._sx, self._sy = 100, 100
        self.seen = []

    def _on_move(self, event):
        self.seen.append((event.x, event.y))


class _Move:
    def __init__(self, x, y):
        self.x, self.y = x, y


def test_a_faster_spin_reports_a_longer_drag(canvas):
    canvas.apply_settings(GateEditorSettings(spin_speed=2.0))
    axes = _Axes()
    canvas._apply_spin_speed(axes)
    axes._on_move(_Move(110, 100))
    assert axes.seen == [(120, 100)]        # 10 px of drag read as 20


def test_a_slower_spin_reports_a_shorter_one(canvas):
    canvas.apply_settings(GateEditorSettings(spin_speed=0.5))
    axes = _Axes()
    canvas._apply_spin_speed(axes)
    axes._on_move(_Move(140, 100))
    assert axes.seen == [(120, 100)]


def test_the_default_speed_wraps_nothing(canvas):
    """No wrap at all at 1.0: matplotlib's own rotation, untouched."""
    canvas.apply_settings(GateEditorSettings(spin_speed=1.0))
    axes = _Axes()
    canvas._apply_spin_speed(axes)
    # A bound method is a fresh object on every access, so identity says
    # nothing; the marker the wrapper sets is what does.
    assert not getattr(axes._on_move, "_spacr_wrapped", False)


def test_wrapping_twice_does_not_compound(canvas):
    canvas.apply_settings(GateEditorSettings(spin_speed=2.0))
    axes = _Axes()
    canvas._apply_spin_speed(axes)
    canvas._apply_spin_speed(axes)
    axes._on_move(_Move(110, 100))
    assert axes.seen == [(120, 100)]        # not 140


def test_a_matplotlib_without_the_hook_leaves_the_speed_alone(canvas):
    """A setting not taking effect, rather than a volume that will not turn."""
    class _NoHook:
        pass

    canvas.apply_settings(GateEditorSettings(spin_speed=2.0))
    canvas._apply_spin_speed(_NoHook())      # the assertion is no raise


def test_the_wrapped_move_still_reaches_matplotlib(canvas):
    canvas.apply_settings(GateEditorSettings(spin_speed=3.0))
    axes = _Axes()
    canvas._apply_spin_speed(axes)
    axes._on_move(_Move(101, 99))
    assert len(axes.seen) == 1
