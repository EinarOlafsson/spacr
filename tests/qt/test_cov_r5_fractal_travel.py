"""The backdrop's refusals, its kernels, and the canvas it cannot always have.

`fractal_travel` is two renderers wrapped in a great deal of declining to
fail: a pointer that answers even when the widget it was handed has been
freed, kernels that saturate rather than tear when the pointer is dragged
across them, a worker that must not shout at a widget Qt has already
deleted, and a GPU canvas that has to give up during CONSTRUCTION so the
factory can fall back.

Three techniques are used here rather than hoping the machine cooperates.

* The Numba kernels are driven through ``.py_func``. A jitted body is not
  Python at run time, so the arithmetic that decides what the picture looks
  like is otherwise never executed as source at all -- and the saturation
  rules inside `_orbit_sample` are the difference between a pointer that
  bends the plane and one that turns it inside out.

* The GPU canvas is built against a STAND-IN ``vispy``. `QT_QPA_PLATFORM=
  offscreen` prints "QOpenGLWidget is not supported on this platform" and
  then behaves in whatever way the local driver behaves, which is not
  something a test can assert against; the same reasoning as
  `test_cov_gpu_backdrop_does_not_storm`, one step further. Everything below
  ``gloo.Program`` is somebody else's code and everything above it is what
  this module does with it, so that is where the seam goes.

* The CPU widget is given a cheap engine before it is built, by replacing
  the module-level `OrbitEngine` the builder looks up. The frame loop is
  then real -- a real QThread, real signals, real timers -- while a frame
  costs a memset instead of half a second of shading.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint                              # noqa: E402
from PySide6.QtWidgets import QWidget                          # noqa: E402

from spacr.qt import preferences as P                          # noqa: E402
from spacr.qt.widgets import fractal_travel as F               # noqa: E402


# =========================================================================
# The pointer, which is never allowed to raise
# =========================================================================


class _StubWidget:
    """Just enough widget for `Pointer.sample` to read a position from.

    A real one cannot be used: the position comes from the global
    `QCursor.pos()`, so where the pointer lands relative to the widget is
    whatever the session's cursor happens to be doing. Deciding
    `mapFromGlobal` here is what makes "inside" and "outside" assertable.
    """

    def __init__(self, point, visible=True, width=200, height=100):
        self._point = point
        self._visible = visible
        self._width = width
        self._height = height

    def isVisible(self):
        return self._visible

    def mapFromGlobal(self, _global):
        return self._point

    def width(self):
        return self._width

    def height(self):
        return self._height


def test_a_widget_that_is_not_on_screen_puts_the_pointer_outside(qapp):
    """Nowhere to be relative to is not "at the centre"."""
    pointer = F.Pointer()
    assert pointer.sample(_StubWidget(QPoint(100, 50))).inside is True

    assert pointer.sample(None) is pointer
    assert pointer.inside is False
    hidden = _StubWidget(QPoint(100, 50), visible=False)
    assert pointer.sample(hidden) is pointer
    assert pointer.inside is False


def test_a_pointer_off_the_widget_lets_go_instead_of_pulling_at_the_edge(qapp):
    """Outside, pull and push DECAY -- they are not merely reported as off.

    Snapping them to zero would let go in one frame, and holding them would
    tug the pattern toward an edge the pointer is nowhere near.
    """
    pointer = F.Pointer()
    centre = _StubWidget(QPoint(100, 50))
    for _ in range(40):
        pointer.sample(centre)
    held = pointer.pull
    assert held > 0.5, held

    pointer.sample(_StubWidget(QPoint(-40, 50)))
    assert pointer.inside is False
    assert pointer.pull == pytest.approx(held - 0.08)
    assert pointer.push == 0.0


def test_a_pointer_asked_about_a_freed_widget_answers_outside(qapp):
    """It is sampled from a render tick; raising there ends the process."""
    class _Freed:
        def isVisible(self):
            raise RuntimeError(
                "Internal C++ object (QWidget) already deleted.")

    pointer = F.Pointer()
    assert pointer.sample(_StubWidget(QPoint(100, 50))).inside is True
    assert pointer.sample(_Freed()) is pointer
    assert pointer.inside is False


# =========================================================================
# How many cores the backdrop is allowed to take
# =========================================================================


def test_a_numba_that_will_not_say_how_many_threads_is_not_fatal(monkeypatch):
    """`NUMBA_NUM_THREADS` is read defensively; a junk value must not stop
    the backdrop being built."""
    class _Junk:
        NUMBA_NUM_THREADS = "as many as you like"

    class _Four:
        NUMBA_NUM_THREADS = 4

    hardware = F.HardwareProfile(logical_cpus=32)
    monkeypatch.setattr(F, "numba_config", _Four)
    capped = F.resolved_cpu_threads(F.Settings(), hardware)
    monkeypatch.setattr(F, "numba_config", _Junk)
    unreadable = F.resolved_cpu_threads(F.Settings(), hardware)

    # Numba's own cap is honoured when it can be read, and ignored when it
    # cannot -- rather than the whole call failing.
    assert capped == 3
    assert unreadable == 19


def test_without_numbas_cap_the_machines_own_count_decides(monkeypatch):
    monkeypatch.setattr(F, "numba_config", None)
    assert F.resolved_cpu_threads(
        F.Settings(), F.HardwareProfile(logical_cpus=8)) == 6


def test_a_small_machine_still_keeps_a_core_for_the_application(monkeypatch):
    """Six or fewer gives one back; two or fewer renders single-threaded."""
    monkeypatch.setattr(F, "numba_config", None)
    assert F.resolved_cpu_threads(
        F.Settings(), F.HardwareProfile(logical_cpus=4)) == 3
    assert F.resolved_cpu_threads(
        F.Settings(), F.HardwareProfile(logical_cpus=6)) == 5
    assert F.resolved_cpu_threads(
        F.Settings(), F.HardwareProfile(logical_cpus=2)) == 1


# =========================================================================
# Which renderer this machine gets
# =========================================================================


def test_an_unrecognised_platform_plugin_is_given_the_benefit_of_the_doubt(
        monkeypatch):
    """`eglfs` is a real GL platform; the list below it is not exhaustive."""
    monkeypatch.delenv("SPACR_NO_GL", raising=False)
    monkeypatch.setenv("QT_QPA_PLATFORM", "eglfs")
    assert F.platform_can_do_opengl() is True
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    assert F.platform_can_do_opengl() is False


def test_a_gl_platform_reports_whether_vispy_is_importable(monkeypatch):
    """And a broken import system counts as no GPU, not as a crash."""
    import importlib.util

    pytest.importorskip("vispy")
    monkeypatch.setattr(F, "platform_can_do_opengl", lambda: True)
    assert F.gpu_is_available() is True

    def _broken(_name):
        raise ValueError("a meta path finder went wrong")

    monkeypatch.setattr(importlib.util, "find_spec", _broken)
    assert F.gpu_is_available() is False


def test_an_explicit_backend_is_taken_at_its_word(monkeypatch):
    """Only `auto` asks; `gpu` and `cpu` are answers already."""
    monkeypatch.setattr(F, "gpu_is_available", lambda: False)
    assert F.resolve_backend("cpu") == "cpu"
    assert F.resolve_backend("gpu") == "gpu"
    assert F.resolve_backend("auto") == "cpu"


# =========================================================================
# The Numba kernels, driven as Python
# =========================================================================


@pytest.fixture
def orbit_sample(monkeypatch):
    """The Python `_orbit_sample` was compiled from, and its helpers with it.

    The kernel calls `_fast_sin`/`_fast_cos` by module-global name, so
    patching the globals is what makes the WHOLE thing run as Python --
    otherwise the inner calls go back through the dispatcher and compile.
    """
    pytest.importorskip("numba")
    sample = F._orbit_sample.py_func
    monkeypatch.setattr(F, "_fast_sin", F._fast_sin.py_func)
    monkeypatch.setattr(F, "_fast_cos", F._fast_cos.py_func)
    monkeypatch.setattr(F, "_orbit_sample", sample)
    return sample


def test_the_fast_trig_stays_bounded_and_tracks_the_real_thing(monkeypatch):
    """A dozen calls per pixel per iteration is why it is approximated; a
    sine that drifted outside -1..1 would blow the colour out."""
    import math

    pytest.importorskip("numba")
    sine = F._fast_sin.py_func
    cosine = F._fast_cos.py_func
    monkeypatch.setattr(F, "_fast_sin", sine)
    for step in range(-60, 61):
        angle = step * 0.37
        assert -1.0001 <= sine(angle) <= 1.0001
        assert sine(angle) == pytest.approx(math.sin(angle), abs=0.002)
        assert cosine(angle) == pytest.approx(math.cos(angle), abs=0.002)


def test_the_orbit_kernel_answers_a_colour_for_every_pixel(orbit_sample):
    sample = orbit_sample
    seen = set()
    for py in range(0, 48, 6):
        for px in range(0, 64, 6):
            red, green, blue = sample(px, py, 64, 48, 3.0, 4.0, 1.5, 5,
                                      0.0, 0.0, 0.0, 0.0)
            assert 0 <= red <= 255 and 0 <= green <= 255 and 0 <= blue <= 255
            seen.add((red, green, blue))
    # The fold has to produce a picture, not a wash: a constant kernel would
    # give one colour for the whole grid.
    assert len(seen) > 8


def test_the_pointer_bends_the_plane_and_then_stops_bending_it(orbit_sample):
    """The 1/r falloff saturates on both sides, which is what keeps a
    pointer resting on a pixel from turning the neighbourhood inside out."""
    sample = orbit_sample
    px, py, width, height = 10, 10, 64, 48
    x = (2.0 * px - width) / min(width, height)
    y = (height - 2.0 * py) / min(width, height)
    near = (x + 0.1, y + 0.1)
    far = (x + 1.0, y + 1.0)

    def at(pointer, pull, push):
        return sample(px, py, width, height, 0.0, 4.0, 1.5, 5,
                      pointer[0], pointer[1], pull, push)

    untouched = at(near, 0.0, 0.0)
    # Well inside the reach, the pull moves the sample...
    assert at(near, 1.0, 0.0) != untouched
    # ...but not five times as far when asked five times as hard.
    assert at(near, 5.0, 0.0) == at(near, 1.0, 0.0)
    # The shove saturates the same way, on its own side of zero.
    assert at(near, 0.0, 1.0) != untouched
    assert at(near, 0.0, 5.0) == at(near, 0.0, 1.0)
    assert at(near, 0.0, 1.0) != at(near, 1.0, 0.0)
    # Far away and gentle it is NOT saturated: doubling the pull there does
    # change the answer, which is what makes the clamp above meaningful.
    assert at(far, 0.02, 0.0) != at(far, 0.01, 0.0)


def test_the_frame_kernel_lays_the_samples_out_at_the_jittered_positions(
        orbit_sample):
    """`_render_into` is the pixel loop and nothing else -- so every texel
    must equal the sample taken at that pixel plus the frame's jitter."""
    sample = orbit_sample
    render = F._render_into.py_func
    output = np.zeros((6, 8, 3), dtype=np.uint8)
    jitter_x, jitter_y = F.JITTERS[2]
    render(output, 1.5, 4.0, 1.5, 5, jitter_x, jitter_y, 0.0, 0.0, 0.0, 0.0)

    for y in range(6):
        for x in range(8):
            expected = sample(x + jitter_x, y + jitter_y, 8, 6,
                              1.5, 4.0, 1.5, 5, 0.0, 0.0, 0.0, 0.0)
            assert tuple(int(v) for v in output[y, x]) == expected
    assert output.std() > 0


def test_the_temporal_blend_weights_the_newest_frame_most(orbit_sample):
    """0.62/0.22/0.10/0.06 over the ring, indexed BACKWARDS from `newest`.

    Getting the ring order wrong would antialias against the future.
    """
    blend = F._blend_temporal.py_func
    ring = np.zeros((4, 2, 3, 3), dtype=np.uint8)
    ring[0] = 200        # newest
    ring[3] = 100        # newest - 1
    ring[2] = 40         # newest - 2
    ring[1] = 10         # newest - 3
    output = np.zeros((2, 3, 3), dtype=np.uint8)
    blend(ring, output, 0)

    expected = int(0.62 * 200 + 0.22 * 100 + 0.10 * 40 + 0.06 * 10)
    assert (output == expected).all()
    assert expected == 150


# =========================================================================
# Joining the render thread when Qt frees the widget
# =========================================================================


class _DestroyedSignal:
    def __init__(self, refuses=False):
        self.handler = None
        self._refuses = refuses

    def connect(self, handler):
        if self._refuses:
            raise RuntimeError("this object has no `destroyed` to connect to")
        self.handler = handler


class _WidgetWithDestroyed:
    def __init__(self, refuses=False):
        self.destroyed = _DestroyedSignal(refuses)


class _Thread:
    def __init__(self, refuses=False):
        self.quits = 0
        self.waited = None
        self._refuses = refuses

    def quit(self):
        self.quits += 1
        if self._refuses:
            raise RuntimeError(
                "Internal C++ object (QThread) already deleted.")

    def wait(self, milliseconds):
        self.waited = milliseconds
        return True


def test_a_thread_that_cannot_be_joined_does_not_raise_out_of_destruction():
    """The handler runs while Qt is tearing the widget down; an exception
    there is a second crash on top of the one it exists to prevent."""
    joined = _WidgetWithDestroyed()
    good = _Thread()
    F._join_on_destroy(joined, good)
    joined.destroyed.handler()
    assert (good.quits, good.waited) == (1, 5000)

    refusing = _WidgetWithDestroyed()
    gone = _Thread(refuses=True)
    F._join_on_destroy(refusing, gone)
    refusing.destroyed.handler()               # must not raise
    assert gone.quits == 1
    assert gone.waited is None                 # it never got that far


def test_a_widget_with_no_destroyed_signal_is_simply_not_watched():
    connectable = _WidgetWithDestroyed()
    F._join_on_destroy(connectable, _Thread())
    assert connectable.destroyed.handler is not None

    refusing = _WidgetWithDestroyed(refuses=True)
    F._join_on_destroy(refusing, _Thread())    # must not raise
    assert refusing.destroyed.handler is None


# =========================================================================
# The CPU widget
# =========================================================================


class _CheapEngine:
    """Answers a frame instantly, so the real loop can be driven in a test.

    Substituted for `OrbitEngine` before the widget is built: the builder
    reads the module global, so this is the seam that does not require
    reaching inside the widget afterwards and racing its thread.
    """

    def __init__(self, thread_count):
        self.thread_count = thread_count
        self.calls = []
        self.explode = None

    def render(self, width, height, t, speed, dream, iterations,
               pointer_x=0.0, pointer_y=0.0, pull=0.0, push=0.0):
        self.calls.append(dict(width=width, height=height, t=t, speed=speed,
                               dream=dream, iterations=iterations,
                               pointer_x=pointer_x, pointer_y=pointer_y,
                               pull=pull, push=push))
        if self.explode is not None:
            raise self.explode
        return np.full((height, width, 3), 7, dtype=np.uint8)


@pytest.fixture
def cpu_widget(qapp, monkeypatch):
    """A live `CpuFractalWidget` whose frames cost nothing."""
    pytest.importorskip("numba")
    monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)
    widget = F._make_cpu_widget(F.Settings(pattern="orbit", backend="cpu"),
                                F.RuntimeControls(),
                                F.HardwareProfile(logical_cpus=4))
    yield widget
    widget.shutdown()
    widget.deleteLater()


def test_the_cpu_backend_says_it_needs_numba(qapp, monkeypatch):
    """Without it `_render_into` raises per frame instead of once, at build."""
    monkeypatch.setattr(F, "njit", None)
    with pytest.raises(RuntimeError, match="numba"):
        F._make_cpu_widget(F.Settings(backend="cpu"), F.RuntimeControls(),
                           F.HardwareProfile(logical_cpus=4))


def test_the_worker_unpacks_the_request_and_hands_the_frame_back(cpu_widget):
    """The request is a plain dict crossing a thread boundary, so the
    mapping from its keys to the engine's positional arguments is the
    contract -- and the pointer keys have to default when they are absent."""
    # DIRECT, because the worker lives on the render thread: an auto
    # connection to a plain callable would queue the delivery onto that
    # thread's event loop, which nothing here is spinning.
    from PySide6.QtCore import Qt

    delivered = []
    cpu_widget._worker.frame_ready.connect(
        lambda frame, seconds: delivered.append((frame, seconds)),
        Qt.ConnectionType.DirectConnection)
    cpu_widget._worker.render({"width": 8, "height": 6, "t": 2.5,
                               "speed": 3.0, "dream": 1.25, "iterations": 5})

    asked = cpu_widget._worker.engine.calls[-1]
    assert asked["width"] == 8 and asked["height"] == 6
    assert (asked["t"], asked["speed"], asked["dream"]) == (2.5, 3.0, 1.25)
    assert (asked["pointer_x"], asked["pointer_y"]) == (0.0, 0.0)
    assert (asked["pull"], asked["push"]) == (0.0, 0.0)
    frame, seconds = delivered[-1]
    assert frame.shape == (6, 8, 3)
    assert seconds >= 0.0


def test_a_frame_that_cannot_be_shaded_is_reported_rather_than_thrown(
        cpu_widget):
    """An exception escaping a slot on a QThread ends the process."""
    from PySide6.QtCore import Qt

    complaints = []
    cpu_widget._worker.failed.connect(complaints.append,
                                      Qt.ConnectionType.DirectConnection)
    cpu_widget._worker.engine.explode = ValueError("no memory for that frame")
    cpu_widget._worker.render({"width": 8, "height": 6, "t": 0.0,
                               "speed": 3.0, "dream": 1.0, "iterations": 5})

    assert complaints == ["ValueError: no memory for that frame"]
    assert cpu_widget._error == "ValueError: no memory for that frame"


def test_a_frame_that_finishes_after_its_widget_is_gone_says_nothing(
        cpu_widget, caplog):
    """`emit` on a freed sender raises RuntimeError, and the handler that
    then emitted the FAILURE signal raised the same way."""
    class _Signal:
        def __init__(self, error):
            self.error = error
            self.attempts = 0

        def emit(self, *args):
            self.attempts += 1
            if self.error is not None:
                raise self.error

    say = cpu_widget._worker._say_something
    live = _Signal(None)
    say(live, "a frame")
    assert live.attempts == 1

    freed = _Signal(RuntimeError("Signal source has been deleted"))
    say(freed, "a frame")                      # must not raise
    assert freed.attempts == 1

    with caplog.at_level("DEBUG", logger=F.LOG.name):
        odd = _Signal(TypeError("wrong argument types"))
        say(odd, "a frame")                    # must not raise either
    assert "could not deliver a frame" in caplog.text


def test_pausing_twice_says_it_was_already_paused(cpu_widget):
    """The return value is how a caller knows whether to resume."""
    assert cpu_widget.pause() is True
    assert cpu_widget.pause() is False
    assert cpu_widget.is_paused() is True
    assert cpu_widget.resume() is True
    assert cpu_widget.resume() is False
    assert cpu_widget.is_paused() is False
    assert cpu_widget.set_animating(False) is True
    assert cpu_widget.set_animating(False) is False


def test_a_stopped_widget_does_not_resume(cpu_widget):
    cpu_widget.pause()
    cpu_widget.shutdown()
    assert cpu_widget.resume() is False
    assert cpu_widget.is_paused() is True


def test_an_unreadable_render_scale_falls_back_to_native(cpu_widget,
                                                         monkeypatch):
    """A setting that cannot be read must not stop a frame being sized."""
    cpu_widget.resize(1600, 1200)
    monkeypatch.setattr(F, "_render_scale", lambda: 1.0)
    native = cpu_widget._target_size()

    def _broken():
        raise RuntimeError("the settings store is not there")

    monkeypatch.setattr(F, "_render_scale", _broken)
    assert cpu_widget._target_size() == native


def test_a_render_scale_of_zero_leaves_the_patterns_own_budget_alone(
        cpu_widget, monkeypatch):
    """Zero is not "shade nothing": it means the number is not in play, and
    the pattern's own pixel budget decides."""
    cpu_widget.resize(1600, 1200)
    monkeypatch.setattr(F, "_render_scale", lambda: 1.0)
    native = cpu_widget._target_size()
    monkeypatch.setattr(F, "_render_scale", lambda: 0.0)
    budgeted = cpu_widget._target_size()

    assert budgeted[0] < native[0] and budgeted[1] < native[1]
    assert budgeted[0] * budgeted[1] == pytest.approx(460_000, rel=0.05)


def test_a_paused_widget_asks_for_no_more_frames(cpu_widget):
    requests = []
    cpu_widget.render_requested.connect(requests.append)
    cpu_widget.show()
    try:
        cpu_widget._request_frame()
        assert len(requests) == 1
        assert requests[0]["iterations"] == 5

        cpu_widget.pause()
        cpu_widget._request_frame()
        assert len(requests) == 1
    finally:
        cpu_widget.hide()


def _frame(width=8, height=6, value=7):
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_a_frame_that_arrives_after_shutdown_is_dropped(cpu_widget):
    """A frame can finish on the render thread after the widget has stopped;
    painting it would hand QImage a buffer nothing is keeping alive."""
    cpu_widget._accept_frame(_frame(), 0.01)
    shown = cpu_widget._image_array
    assert cpu_widget._image is not None

    cpu_widget.shutdown()
    cpu_widget._accept_frame(_frame(4, 4, value=200), 0.02)
    assert cpu_widget._image_array is shown


def test_a_paused_widget_shows_the_last_frame_and_asks_for_no_more(cpu_widget):
    """Pause leaves the picture up. It stops the LOOP, not the display."""
    cpu_widget._timer.stop()
    cpu_widget._accept_frame(_frame(8, 6), 0.001)
    assert cpu_widget._timer.isActive() is True

    cpu_widget.pause()
    assert cpu_widget._timer.isActive() is False
    cpu_widget._accept_frame(_frame(12, 10), 0.001)
    assert cpu_widget._image.width() == 12          # still displayed
    assert cpu_widget._timer.isActive() is False    # but nothing scheduled


def test_the_render_scale_follows_the_time_a_frame_actually_took(cpu_widget):
    """The backdrop thins itself out rather than running late, and recovers
    when the machine frees up -- but only within bounds, and only every two
    dozen frames, so it drifts instead of pumping."""
    def settle(seconds):
        # Set directly: `_adapt_resolution` reads a 24-frame counter and a
        # smoothed frame time, and waiting for two dozen real frames to
        # average out to a chosen number is not something a test can do.
        cpu_widget._frames = 23
        cpu_widget._render_ema = seconds
        cpu_widget._accept_frame(_frame(), seconds)
        return cpu_widget._adaptive_scale

    assert cpu_widget._adaptive_scale == 1.0
    slow = settle(0.1)                     # way over the 33 ms period
    assert slow == pytest.approx(0.82)
    steady = settle(0.020)                 # inside the dead band
    assert steady == slow
    fast = settle(0.002)                   # room to spare
    assert fast > steady


def test_a_failure_is_shown_and_retried_unless_the_widget_is_paused(
        cpu_widget):
    """It has to keep trying: a transient failure that stopped the backdrop
    for good would leave a black rectangle behind the whole application."""
    cpu_widget._timer.stop()
    cpu_widget._busy = True
    cpu_widget._on_failure("ValueError: no memory for that frame")
    assert "no memory for that frame" in cpu_widget.stats_text()
    assert cpu_widget._busy is False
    assert cpu_widget._timer.isActive() is True

    cpu_widget.pause()
    cpu_widget._on_failure("ValueError: and again")
    assert "and again" in cpu_widget.stats_text()
    assert cpu_widget._timer.isActive() is False


def test_the_stats_line_says_which_of_three_things_it_is_doing(cpu_widget):
    assert F.VERSION in cpu_widget.stats_text()
    assert "compiling" in cpu_widget.stats_text()

    cpu_widget._accept_frame(_frame(), 0.0125)
    assert "12.5 ms frame" in cpu_widget.stats_text()

    cpu_widget.pause()
    assert "paused for a run" in cpu_widget.stats_text()


def test_closing_the_widget_stops_it_for_good(cpu_widget):
    assert cpu_widget._stopped is False
    cpu_widget.close()
    assert cpu_widget._stopped is True
    assert cpu_widget._timer.isActive() is False


def test_a_resize_asks_for_a_fresh_frame_unless_there_is_no_point(cpu_widget):
    """The frame is sized to the widget, so an old one is the wrong shape --
    but a paused or busy widget already has a frame coming or none wanted."""
    from PySide6.QtCore import QSize
    from PySide6.QtGui import QResizeEvent
    from PySide6.QtWidgets import QApplication

    def resize_to(width, height):
        old = QSize(cpu_widget.width(), cpu_widget.height())
        cpu_widget.resize(width, height)
        QApplication.sendEvent(cpu_widget,
                               QResizeEvent(QSize(width, height), old))

    cpu_widget._timer.stop()
    resize_to(700, 500)
    assert cpu_widget._timer.isActive() is True

    cpu_widget.pause()
    assert cpu_widget._timer.isActive() is False
    resize_to(640, 480)
    assert cpu_widget._timer.isActive() is False


# =========================================================================
# The live backdrops a key press has to reach
# =========================================================================


_SAVED_CONTROLS = {
    "speed": 9.0, "dream": 2.0, "variable_speed": True,
    "speed_min": 1.0, "speed_max": 7.0, "speed_period": 30.0,
    "pointer_gravity": False, "pointer_size": 0.5,
    "pointer_strength": 1.5, "zoom_rate": 2.0,
}


class _DeafControls:
    """A `RuntimeControls` that has gone away underneath the registry."""

    @property
    def restart_token(self):
        raise AttributeError("restart_token")

    @property
    def zoom_rate(self):
        raise AttributeError("zoom_rate")

    def __setattr__(self, name, value):
        raise AttributeError(name)


@pytest.fixture
def live_controls(monkeypatch):
    """`_LIVE_CONTROLS` is module state shared by every backdrop."""
    monkeypatch.setattr(F, "_LIVE_CONTROLS", [])
    return F._LIVE_CONTROLS


def test_saved_settings_reach_a_running_backdrop_unless_they_cannot_be_read(
        live_controls, monkeypatch):
    """The backdrop keeps the controls it was built with, so saving
    Preferences has to push into them; a store that will not open must leave
    what is on screen alone rather than half-updating it."""
    controls = F.RuntimeControls(speed=4.0)
    live_controls.append(controls)
    monkeypatch.setattr(P, "get_fractal_settings",
                        lambda: dict(_SAVED_CONTROLS))
    assert F.apply_saved_controls() == 1
    assert controls.speed == 9.0
    assert controls.follow_pointer is False
    assert controls.zoom_rate == 2.0

    def _broken():
        raise RuntimeError("the settings store is not there")

    monkeypatch.setattr(P, "get_fractal_settings", _broken)
    assert F.apply_saved_controls() == 0
    assert controls.speed == 9.0


def test_one_dead_backdrop_does_not_stop_the_live_one_being_updated(
        live_controls, monkeypatch):
    """The list is only trimmed on build, so it can hold objects that have
    stopped answering."""
    controls = F.RuntimeControls(speed=4.0)
    live_controls.extend([_DeafControls(), controls])
    monkeypatch.setattr(P, "get_fractal_settings",
                        lambda: dict(_SAVED_CONTROLS))

    assert F.apply_saved_controls() == 1
    assert controls.speed == 9.0


def test_a_key_press_that_one_backdrop_refuses_still_reaches_the_rest(
        live_controls):
    good = F.RuntimeControls(zoom_rate=1.0)
    live_controls.extend([_DeafControls(), good])

    F.restart_the_dive()
    assert good.restart_token == 1

    rate = F.nudge_zoom_rate(1)
    assert good.zoom_rate == pytest.approx(F.ZOOM_STEP)
    assert rate == pytest.approx(F.ZOOM_STEP)


def test_the_render_scale_is_read_from_the_store_or_assumed_native(
        monkeypatch):
    monkeypatch.setattr(P, "get_fractal_settings",
                        lambda: {"render_scale": 0.5})
    assert F._render_scale() == 0.5

    def _broken():
        raise RuntimeError("the settings store is not there")

    monkeypatch.setattr(P, "get_fractal_settings", _broken)
    assert F._render_scale() == 1.0


def test_a_backdrop_is_registered_by_identity_not_by_value(
        live_controls, qapp, monkeypatch):
    """`RuntimeControls` is a dataclass, so `in` compares field by field: a
    new backdrop whose settings matched a previous one was never added, and
    Ctrl+R and the arrow keys then drove an object no canvas was reading."""
    monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)
    settings = F.Settings(pattern="orbit", backend="cpu")
    controls = F.RuntimeControls()
    twin = F.RuntimeControls()
    assert twin == controls and twin is not controls

    built = [F.create_fractal_widget(settings, controls),
             F.create_fractal_widget(settings, controls),
             F.create_fractal_widget(settings, twin)]
    try:
        assert sum(1 for c in live_controls if c is controls) == 1
        assert sum(1 for c in live_controls if c is twin) == 1
    finally:
        for widget in built:
            widget.shutdown()
            widget.deleteLater()


def test_a_mandelbrot_with_no_working_gpu_is_drawn_as_the_orbit_fold(
        live_controls, qapp, monkeypatch, caplog):
    """Handing it to the CPU builder silently produced the orbit fold,
    because that is what its final `else` does -- so the user chose
    Mandelbrot and got something else with nothing to say why."""
    monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)
    monkeypatch.setattr(F, "platform_can_do_opengl", lambda: True)
    monkeypatch.setattr(F, "gpu_is_available", lambda: True)

    def refuse(*_args, **_kwargs):
        raise F.GpuBackendError("the fractal shaders do not compile here")

    monkeypatch.setattr(F, "_make_gpu_widget", refuse)
    with caplog.at_level("WARNING", logger=F.LOG.name):
        widget = F.create_fractal_widget(
            F.Settings(pattern="mandelbrot", backend="gpu"))
    try:
        assert widget.backend_name == "cpu"
        assert F.FALLBACK_PATTERN in widget.stats_text()
        assert "mandelbrot" not in widget.stats_text()
        assert "the fractal shaders do not compile here" in caplog.text
        assert "needs the GPU renderer" in caplog.text
    finally:
        widget.shutdown()
        widget.deleteLater()


# =========================================================================
# The GPU canvas, built against a stand-in vispy
# =========================================================================


#: What the Mandelbrot canvas reads out of the settings store.
#:
#: `max_depth` and `initial_scale` are DELIBERATELY absent: the first has a
#: fallback at the call site and the second has none, so between them the
#: three ways `_mandel_setting` can answer are all exercised by the ordinary
#: build rather than by a special case.
_MANDEL_SAVED = {
    "seconds_per_decade": 2.0,
    "base_iterations": 100,
    "iterations_per_decade": 10.0,
    "max_iterations": 64,
    "precision_digits": 30,
    "steering_strength": 0.09,
    "steering_interval_decades": 0.4,
    "steering_duration": 3.8,
    "candidate_count": 4,
}


class _StandInOrbit:
    """A `ReferenceOrbit` that costs nothing to build.

    The real one iterates a couple of thousand points at 320 decimal digits,
    which is seconds -- that is why the canvas builds it on a thread at all.
    """

    def __init__(self, max_iter=2200, digits=320, center=None):
        self.max_iter = int(max_iter)
        self.digits = int(digits)
        self.center = center
        self.packed = np.zeros((2, self.max_iter + 1, 4), dtype=np.float32)


class _FixedPointer:
    """A pointer at a known place, since the real one reads `QCursor.pos()`."""

    def __init__(self, x=0.3, y=-0.2, pull=0.7, push=0.4, error=None):
        self.x = x
        self.y = y
        self.pull = pull
        self.push = push
        self.drag_x = 0.0
        self.drag_y = 0.0
        self.error = error
        self.samples = 0

    def sample(self, _widget, _size=1.0, _strength=1.0):
        self.samples += 1
        if self.error is not None:
            raise self.error
        return self


@pytest.fixture
def stand_in_vispy(qapp, monkeypatch):
    """A `vispy` made entirely of Python, for the length of one test.

    Under `QT_QPA_PLATFORM=offscreen` Qt says "QOpenGLWidget is not supported
    on this platform" and then does whatever the local driver does, which is
    not something to assert against. Everything below `gloo.Program` is
    vispy's code; everything above it is this module's, and that is where
    the seam goes.
    """
    world = types.SimpleNamespace(
        programs=[], textures=[], timers=[], canvases=[], viewports=[],
        states=[], used_app=[], texture_error=None, finish_error=None)

    class _Program(dict):
        def __init__(self, vertex, fragment):
            super().__init__()
            self.vertex = vertex
            self.fragment = fragment
            self.draws = 0
            self.set_error = None
            self.draw_error = None
            world.programs.append(self)

        def __setitem__(self, name, value):
            if self.set_error is not None:
                raise self.set_error
            super().__setitem__(name, value)

        def draw(self, mode):
            self.draws += 1
            if self.draw_error is not None:
                raise self.draw_error

    class _Texture2D:
        def __init__(self, data, **options):
            if world.texture_error is not None:
                raise world.texture_error
            self.data = data
            self.options = options
            world.textures.append(self)

    class _Timer:
        def __init__(self, interval=None, connect=None, start=False):
            self.interval = interval
            self.tick = connect
            self.running = bool(start)
            self.stop_error = None
            world.timers.append(self)

        def stop(self):
            if self.stop_error is not None:
                raise self.stop_error
            self.running = False

    class _Canvas:
        def __init__(self, keys=None, size=(1, 1), show=False, **_options):
            self.native = QWidget()
            self.native.resize(*size)
            self.physical_size = size
            self.currents = 0
            self.updates = 0
            self.closed = False
            self.close_error = None
            world.canvases.append(self)

        def set_current(self):
            self.currents += 1

        def update(self):
            self.updates += 1

        def close(self):
            if self.close_error is not None:
                raise self.close_error
            self.closed = True

    def _finish():
        if world.finish_error is not None:
            raise world.finish_error

    gloo = types.SimpleNamespace(
        Program=_Program, Texture2D=_Texture2D,
        set_state=lambda **options: world.states.append(options),
        set_viewport=lambda *box: world.viewports.append(box),
        gl=types.SimpleNamespace(glFinish=_finish))
    app_module = types.ModuleType("vispy.app")
    app_module.Canvas = _Canvas
    app_module.Timer = _Timer
    app_module.use_app = world.used_app.append
    vispy = types.ModuleType("vispy")
    vispy.app = app_module
    vispy.gloo = gloo

    monkeypatch.setitem(sys.modules, "vispy", vispy)
    monkeypatch.setitem(sys.modules, "vispy.app", app_module)
    return world


@pytest.fixture
def gpu_backdrop(stand_in_vispy):
    """Build GPU backdrops and shut them all down afterwards."""
    built = []

    def build(pattern="orbit", controls=None, **fields):
        widget = F._make_gpu_widget(
            F.Settings(pattern=pattern, backend="gpu", **fields),
            controls if controls is not None else F.RuntimeControls(),
            F.HardwareProfile(logical_cpus=4))
        built.append(widget)
        return widget

    yield build
    for widget in built:
        widget.shutdown()
        widget.deleteLater()


def test_a_vispy_that_will_not_import_is_a_gpu_backend_error(monkeypatch):
    """`create_fractal_widget` catches this one and falls back; anything it
    does not recognise would come out of the constructor instead."""
    monkeypatch.setitem(sys.modules, "vispy", None)
    with pytest.raises(F.GpuBackendError):
        F._make_gpu_widget(F.Settings(backend="gpu"), F.RuntimeControls(),
                           F.HardwareProfile(logical_cpus=4))


def test_the_gpu_canvas_asks_vispy_for_the_binding_the_application_uses(
        gpu_backdrop, stand_in_vispy):
    """Two Qt bindings in one process do not raise -- they segfault."""
    gpu_backdrop("orbit")
    assert stand_in_vispy.used_app == ["pyside6"]


def test_the_preloader_lock_is_found_or_the_backdrop_manages_without_it(
        monkeypatch):
    """Building a GL context while the preloader brings CUDA up is exactly
    the concurrent initialisation the lock exists to prevent -- but this
    widget is usable with no application around it at all."""
    from spacr.qt import app as qt_app

    assert F._heavy_import_lock() is qt_app.HEAVY_IMPORT_LOCK
    monkeypatch.delattr(qt_app, "HEAVY_IMPORT_LOCK")
    assert F._heavy_import_lock() is None


def test_the_canvas_is_built_with_or_without_the_lock(gpu_backdrop,
                                                      monkeypatch,
                                                      stand_in_vispy):
    gpu_backdrop("orbit")
    assert len(stand_in_vispy.canvases) == 1

    monkeypatch.setattr(F, "_heavy_import_lock", lambda: None)
    gpu_backdrop("orbit")
    assert len(stand_in_vispy.canvases) == 2


def test_each_pattern_gets_its_own_shader_and_its_own_detail_budget(
        gpu_backdrop, stand_in_vispy, monkeypatch):
    """One uniform update, four shaders. Sharing one set of numbers would
    make one of them either wasteful or unusable."""
    from spacr.qt.widgets import (fractal_cascade, fractal_mandelbrot,
                                  fractal_space)

    monkeypatch.setattr(fractal_mandelbrot, "ReferenceOrbit", _StandInOrbit)
    monkeypatch.setattr(P, "get_fractal_settings", lambda: dict(_MANDEL_SAVED))

    shaders = {}
    details = {}
    for pattern in ("orbit", "cascade", "space", "mandelbrot"):
        canvas = gpu_backdrop(pattern)._canvas
        shaders[pattern] = canvas._program.fragment
        details[pattern] = canvas._detail

    assert shaders["orbit"] == F.FRAGMENT_SHADER
    assert shaders["cascade"] == fractal_cascade.FRAGMENT_SHADER
    assert shaders["space"] == fractal_space.FRAGMENT_SHADER
    assert shaders["mandelbrot"] == fractal_mandelbrot.FRAGMENT_SHADER
    assert len(set(shaders.values())) == 4
    # The cheap scene carries the smallest budget and the folds the largest.
    assert details == {"orbit": 6, "cascade": 5, "space": 4, "mandelbrot": 6}


def test_only_the_uniforms_a_shader_declares_are_handed_to_it(gpu_backdrop):
    """vispy warns once per DRAW for a name it cannot find, so GPU space
    printed "Value provided for 'u_dream'" sixty times a second."""
    starry = gpu_backdrop("space")._canvas._program
    folded = gpu_backdrop("orbit")._canvas._program

    # A star field has nothing to warp, so its shader never declares it...
    assert "u_dream" not in starry
    # ...while the fold's does, and is given it.
    assert "u_dream" in folded
    # Both are handed what they DO declare.
    assert "u_time" in starry and "u_time" in folded
    assert tuple(starry["u_resolution"]) == (1200, 760)


def test_the_pointer_reaches_the_gpu_but_never_slides_the_deep_zoom(
        gpu_backdrop, monkeypatch):
    """The three fields can be warped toward a point and look right doing
    it. A deep zoom is a CAMERA: pulling its coordinates toward wherever the
    mouse rests slides the picture continuously, which reads as the image
    drifting away from you rather than as anything you did."""
    from spacr.qt.widgets import fractal_mandelbrot as M

    monkeypatch.setattr(M, "ReferenceOrbit", _StandInOrbit)
    monkeypatch.setattr(P, "get_fractal_settings", lambda: dict(_MANDEL_SAVED))
    controls = F.RuntimeControls(follow_pointer=True)

    folded = gpu_backdrop("orbit", controls=controls)._canvas
    # The real pointer reads the global cursor, so where it is cannot be
    # asserted; a fixed one can.
    folded._pointer = _FixedPointer()
    assert folded._pointer_state() == (0.3, -0.2, 0.7, 0.4)

    diving = gpu_backdrop("mandelbrot", controls=controls)._canvas
    diving._pointer = _FixedPointer()
    assert diving._pointer_state() == (0.3, -0.2, 0.0, 0.0)

    # Turned off, the kernels take a zero rather than a position.
    controls.follow_pointer = False
    assert folded._pointer_state() == (0.0, 0.0, 0.0, 0.0)

    # And a backdrop that cannot find the mouse still draws.
    controls.follow_pointer = True
    folded._pointer = _FixedPointer(error=RuntimeError("no cursor here"))
    assert folded._pointer_state() == (0.0, 0.0, 0.0, 0.0)


def test_a_resize_moves_the_viewport_to_the_new_physical_size(gpu_backdrop,
                                                              stand_in_vispy):
    canvas = gpu_backdrop("orbit")._canvas
    canvas.physical_size = (640, 480)
    canvas.on_resize(None)
    assert stand_in_vispy.viewports[-1] == (0, 0, 640, 480)


def test_a_draw_that_fails_stops_the_canvas_instead_of_storming(gpu_backdrop,
                                                                caplog):
    """vispy catches whatever a DrawEvent handler raises, logs it and RETRIES
    -- doubling the repeat count -- so one impossible draw fills the
    terminal for ever."""
    canvas = gpu_backdrop("orbit")._canvas
    canvas.on_draw(None)
    assert canvas._program.draws >= 1
    assert canvas._dead is False

    canvas._program.draw_error = RuntimeError("the GL context was lost")
    with caplog.at_level("WARNING", logger=F.LOG.name):
        canvas.on_draw(None)
    assert canvas._dead is True
    assert canvas._timer.running is False
    assert "stopped drawing" in caplog.text

    before = canvas._program.draws
    canvas.on_draw(None)
    assert canvas._program.draws == before


def test_the_detail_follows_the_time_the_gpu_actually_took(gpu_backdrop):
    """Sampled every two seconds, not every frame: `glFinish` is a stall,
    and measuring the picture must not be what makes it late."""
    canvas = gpu_backdrop("orbit")._canvas

    def draw_with(smoothed):
        # `_last_sample` is what makes a draw a BENCHMARK; a test cannot
        # wait two seconds between each of them.
        canvas._last_sample = 0.0
        canvas._render_ema = smoothed
        canvas.on_draw(None)
        return canvas._detail

    assert canvas._detail == 6
    assert draw_with(0.1) == 5              # far over the 16 ms period
    assert draw_with(0.1) == 5              # and never below the floor
    assert draw_with(0.016) == 5            # inside the dead band: no change
    assert draw_with(0.0) == 6              # room to spare
    assert draw_with(0.0) == 7              # one over the base is the ceiling
    assert draw_with(0.0) == 7

    # A draw inside the two-second window is not a benchmark at all.
    drawn = canvas._program.draws
    sampled_at = canvas._last_sample
    canvas.on_draw(None)
    assert canvas._program.draws == drawn + 1
    assert canvas._last_sample == sampled_at


def test_a_gl_that_will_not_finish_still_gets_its_next_benchmark(
        gpu_backdrop, stand_in_vispy):
    """Otherwise `_last_sample` never moves and every frame stalls on a
    `glFinish` that is already failing."""
    canvas = gpu_backdrop("orbit")._canvas
    canvas._last_sample = 0.0
    stand_in_vispy.finish_error = RuntimeError("glFinish is not available")
    canvas.on_draw(None)
    assert canvas._render_ema is None       # never measured
    assert canvas._last_sample > 0.0        # but rescheduled


def test_a_timer_tick_at_a_canvas_qt_has_freed_stops_the_timer(gpu_backdrop):
    """vispy's Timer is not a QTimer and is not destroyed with the widget, so
    it goes on firing at a canvas that is gone -- and vispy's handler
    catches, logs and RETRIES, which is where 2, 4, 8 ... 4096 comes from."""
    canvas = gpu_backdrop("orbit")._canvas
    canvas._on_timer(None)
    assert canvas.updates == 1

    canvas._paused = True
    canvas._on_timer(None)
    assert canvas.updates == 1
    canvas._paused = False

    canvas._program.set_error = RuntimeError(
        "Internal C++ object (Canvas) already deleted.")
    canvas._on_timer(None)
    assert canvas._dead is True
    assert canvas._timer.running is False

    canvas._on_timer(None)                  # dead means dead
    assert canvas.updates == 1


def test_the_timer_can_be_stopped_twice_and_after_deletion(gpu_backdrop):
    canvas = gpu_backdrop("orbit")._canvas
    canvas.stop_timer()
    assert canvas._timer.running is False

    canvas._timer.stop_error = RuntimeError("the timer is gone")
    canvas.stop_timer()                     # must not raise


def test_qt_freeing_the_native_widget_kills_the_canvas(gpu_backdrop):
    """A backdrop is reparented and deleted with its screen, which never
    runs closeEvent."""
    canvas = gpu_backdrop("orbit")._canvas
    assert canvas._dead is False
    canvas.native.destroyed.emit()
    assert canvas._dead is True
    assert canvas._timer.running is False


def test_the_gpu_stats_line_says_which_of_three_things_it_is_doing(
        gpu_backdrop):
    canvas = gpu_backdrop("orbit")._canvas
    assert "measuring" in canvas.stats_text()
    assert "GPU/balanced" in canvas.stats_text()
    assert "detail 6" in canvas.stats_text()

    canvas._render_ema = 0.0125
    assert "12.5 ms GPU" in canvas.stats_text()

    canvas._paused = True
    assert "paused for a run" in canvas.stats_text()


def test_the_gpu_widget_winds_down_and_starts_again(gpu_backdrop):
    widget = gpu_backdrop("orbit")
    assert widget.backend_name == "gpu"
    assert widget.is_paused() is False
    assert widget.pause() is True
    assert widget.pause() is False
    assert widget.is_paused() is True
    assert widget.stats_text() == widget._canvas.stats_text()
    assert widget.resume() is True
    assert widget.resume() is False
    assert widget.set_animating(False) is True
    assert widget.set_animating(True) is True


def test_closing_the_gpu_widget_stops_it_even_when_the_canvas_is_gone(
        gpu_backdrop):
    """Safe to call twice and after Qt has freed it."""
    widget = gpu_backdrop("orbit")
    canvas = widget._canvas
    widget.close()
    assert canvas._dead is True
    assert canvas.closed is True
    assert canvas._timer.running is False

    other = gpu_backdrop("orbit")
    other._canvas.close_error = RuntimeError("already deleted")
    other.shutdown()                        # must not raise
    assert other._canvas._dead is True


# =========================================================================
# The Mandelbrot dive, which is the GPU canvas's own machinery
# =========================================================================


def _until(predicate, seconds=5.0):
    """Wait for a worker thread to land. Returns whether it did."""
    import time as _time

    deadline = _time.perf_counter() + seconds
    while _time.perf_counter() < deadline:
        if predicate():
            return True
        _time.sleep(0.005)
    return bool(predicate())


@pytest.fixture
def mandel(gpu_backdrop, monkeypatch):
    """A Mandelbrot canvas with a settings store and a cheap reference."""
    from spacr.qt.widgets import fractal_mandelbrot as M

    monkeypatch.setattr(M, "ReferenceOrbit", _StandInOrbit)
    saved = dict(_MANDEL_SAVED)
    monkeypatch.setattr(P, "get_fractal_settings", lambda: dict(saved))

    def build(controls=None):
        canvas = gpu_backdrop("mandelbrot", controls=controls)._canvas
        if canvas._orbit_thread is not None:
            canvas._orbit_thread.join(10)
        return canvas

    return types.SimpleNamespace(build=build, saved=saved, module=M)


def test_the_saved_mandelbrot_numbers_are_used_and_the_rest_defaulted(
        mandel, monkeypatch):
    """Every one of the twelve settings the panel offers was collected,
    stored and then ignored -- the published defaults are the FALLBACK now,
    not the answer."""
    built = []

    class _Recording(_StandInOrbit):
        def __init__(self, **fields):
            super().__init__(**fields)
            built.append(self)

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Recording)
    canvas = mandel.build()
    assert (built[0].max_iter, built[0].digits) == (64, 30)
    # `initial_scale` is not in the store and has no fallback at the call
    # site, so the published default is what answers.
    assert float(canvas._program["u_scale"]) == pytest.approx(1.25)

    def _broken():
        raise RuntimeError("the settings store is not there")

    monkeypatch.setattr(P, "get_fractal_settings", _broken)
    mandel.build()
    assert (built[1].max_iter, built[1].digits) == (2200, 320)


def test_a_reference_orbit_that_cannot_be_built_leaves_the_dive_drawing(
        mandel, monkeypatch, caplog):
    """Until it arrives the shader has an all-zero orbit, which renders as
    the flat interior colour rather than as a stall."""
    working = mandel.build()
    assert working._orbit is not None

    class _Refusing:
        def __init__(self, **_fields):
            raise ValueError("mpmath is not installed")

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Refusing)
    with caplog.at_level("ERROR", logger=F.LOG.name):
        broken = mandel.build()
    assert broken._orbit is None
    assert "could not build the reference orbit" in caplog.text


def test_the_orbit_texture_is_seeded_and_a_refusal_is_only_logged(
        mandel, monkeypatch, stand_in_vispy, caplog):
    """vispy warns once per DRAW for a uniform a linked program has never
    been given, and the real orbit takes seconds to iterate."""
    class _Refusing:
        def __init__(self, **_fields):
            raise ValueError("no mpmath")

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Refusing)
    seeded = mandel.build()
    assert seeded._program["u_orbit"].data.shape == (1, 1, 4)

    stand_in_vispy.texture_error = RuntimeError("no float textures here")
    with caplog.at_level("DEBUG", logger=F.LOG.name):
        bare = mandel.build()
    assert "u_orbit" not in bare._program
    assert "could not seed the orbit texture" in caplog.text


def test_the_finished_orbit_is_uploaded_exactly_once(mandel, monkeypatch,
                                                     stand_in_vispy, caplog):
    """`_orbit_uploaded` is the orbit OBJECT, so a rebuilt one is noticed and
    a driver that refuses the format is not asked sixty times a second."""
    class _Refusing:
        def __init__(self, **_fields):
            raise ValueError("no mpmath")

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Refusing)
    canvas = mandel.build()
    before = len(stand_in_vispy.textures)
    canvas._update_uniforms(0.1)
    assert len(stand_in_vispy.textures) == before      # nothing to upload

    orbit = _StandInOrbit(max_iter=4, digits=20)
    canvas._orbit = orbit
    canvas._update_uniforms(0.2)
    assert canvas._program["u_orbit"].data is orbit.packed
    assert len(stand_in_vispy.textures) == before + 1
    canvas._update_uniforms(0.3)
    assert len(stand_in_vispy.textures) == before + 1  # and not again

    stand_in_vispy.texture_error = RuntimeError("rgba32f is not supported")
    canvas._orbit = _StandInOrbit(max_iter=4, digits=20)
    with caplog.at_level("ERROR", logger=F.LOG.name):
        canvas._update_uniforms(0.4)
        canvas._update_uniforms(0.5)
    assert caplog.text.count("could not upload the reference orbit") == 1


def test_only_the_deep_zoom_carries_the_deep_zoom_uniforms(mandel,
                                                           gpu_backdrop):
    """The shared update splices this in unconditionally, so it has to be
    empty for the other three rather than absent."""
    folded = gpu_backdrop("orbit")._canvas
    assert folded._mandelbrot_uniforms(1.0) == {}

    diving = mandel.build()
    assert set(diving._mandelbrot_uniforms(1.0)) == {
        "u_scale", "u_center_offset", "u_depth", "u_orbit_length",
        "u_max_iter"}


def test_the_dive_integrates_its_depth_and_can_back_out_of_it(mandel):
    """Up and Down change the RATE, and a depth derived from elapsed*rate
    would jump backwards the moment the rate was lowered."""
    import time as _time

    controls = F.RuntimeControls(speed=4.0, zoom_rate=1.0)
    canvas = mandel.build(controls=controls)

    # The clock is `time.perf_counter`; a test cannot make a second pass, so
    # the last reading is moved back instead.
    canvas._depth = 5.0
    canvas._zoom_clock = _time.perf_counter() - 1.0
    deeper = float(canvas._mandelbrot_uniforms(0.0)["u_depth"])
    # One second, four units of speed, two seconds a decade: two decades.
    assert deeper == pytest.approx(7.0, abs=0.05)

    controls.zoom_rate = -4.0
    canvas._zoom_clock = _time.perf_counter() - 1.0
    backed_out = float(canvas._mandelbrot_uniforms(0.0)["u_depth"])
    assert backed_out == 0.0                # floored at the surface


def test_a_restart_takes_the_dive_and_the_camera_back_to_the_surface(mandel):
    """A restart that kept the course would begin at the surface already
    pointed a whole descent of steering away from the centre."""
    controls = F.RuntimeControls()
    canvas = mandel.build(controls=controls)
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._mandelbrot_uniforms(0.0)        # makes the camera
    camera = canvas._camera
    camera.centre = (0.25, -0.5)
    camera.step = 7
    canvas._depth = 6.0

    controls.restart_token += 1
    after = canvas._mandelbrot_uniforms(0.0)
    assert float(after["u_depth"]) == 0.0
    assert camera.centre == (0.0, 0.0)
    assert camera.step == 0
    assert canvas._plan is None


def test_the_reference_orbit_caps_the_iteration_budget(mandel):
    """Every pixel perturbs around it, so asking for more iterations than
    the reference has is asking about numbers that are not there."""
    canvas = mandel.build()
    canvas._orbit = None
    uncapped = canvas._mandelbrot_uniforms(0.0)
    assert float(uncapped["u_orbit_length"]) == 1.0
    assert int(uncapped["u_max_iter"]) == 100

    canvas._orbit = _StandInOrbit(max_iter=3, digits=20)
    capped = canvas._mandelbrot_uniforms(0.0)
    assert float(capped["u_orbit_length"]) == 4.0
    assert int(capped["u_max_iter"]) == 3


def test_a_fault_in_the_steering_still_returns_a_whole_frame(mandel,
                                                             monkeypatch,
                                                             caplog):
    """A NameError in the course-plotting once left every uniform unset, so
    the pattern drew nothing at all -- silently."""
    from spacr.qt.widgets.fractal_mandelbrot import SteeringCamera

    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    camera = SteeringCamera()
    camera.centre = (0.125, -0.25)
    canvas._camera = camera

    def _broken(*_args, **_kwargs):
        raise NameError("name 'plot_the_course' is not defined")

    monkeypatch.setattr(canvas, "_steer", _broken)
    with caplog.at_level("ERROR", logger=F.LOG.name):
        values = canvas._mandelbrot_uniforms(0.0)

    assert set(values) == {"u_scale", "u_center_offset", "u_depth",
                           "u_orbit_length", "u_max_iter"}
    assert tuple(float(v) for v in values["u_center_offset"]) == (0.125, -0.25)
    assert "could not steer the dive" in caplog.text


def test_the_camera_is_made_once_and_kept(mandel, monkeypatch):
    """It carries the course; rebuilding it every frame would restart the
    dive's aim sixty times a second."""
    class _Refusing:
        def __init__(self, **_fields):
            raise ValueError("no mpmath")

    monkeypatch.setattr(mandel.module, "ReferenceOrbit", _Refusing)
    canvas = mandel.build()
    # With no reference there is nothing to be pointed at, so there is no
    # camera either and the dive looks straight down.
    assert getattr(canvas, "_camera", None) is None
    nowhere = canvas._mandelbrot_uniforms(0.0)
    assert tuple(float(v) for v in nowhere["u_center_offset"]) == (0.0, 0.0)

    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._mandelbrot_uniforms(0.0)
    camera = canvas._camera
    assert camera is not None
    canvas._mandelbrot_uniforms(0.0)
    assert canvas._camera is camera


def test_dragging_moves_the_camera_and_pulls_the_reference_after_it(mandel):
    """Perturbation measures every pixel as an offset from ONE orbit, so a
    camera that walks away from it takes the picture with it."""
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    pointer = _FixedPointer()
    canvas._pointer = pointer
    canvas._mandelbrot_uniforms(0.0)
    camera = canvas._camera
    camera.target = (0.5, 0.5)
    settled = camera.centre

    pointer.drag_x, pointer.drag_y = 0.5, -0.25
    dragged = canvas._mandelbrot_uniforms(0.0)
    assert camera.centre != settled
    assert tuple(float(v) for v in dragged["u_center_offset"]) == (
        pytest.approx(camera.centre[0]), pytest.approx(camera.centre[1]))
    # Consumed, so a dropped frame does not pan twice...
    assert (pointer.drag_x, pointer.drag_y) == (0.0, 0.0)
    # ...the reference is asked to follow at once...
    assert canvas._refine_due == 0.0
    # ...and the hand wins: the camera stops chasing where it was headed.
    assert camera.target is None


def test_the_fixed_path_stays_pointed_at_the_reference(mandel):
    """Surveying the surface for a "more interesting" point was tried and
    made it worse; only a genuinely special point survives a descent, and
    the reference centre already is one."""
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._mandelbrot_uniforms(0.0)
    canvas._camera.centre = (0.375, -0.125)

    values = canvas._mandelbrot_uniforms(0.0)
    assert tuple(float(v) for v in values["u_center_offset"]) == (
        pytest.approx(0.375), pytest.approx(-0.125))
    assert canvas._camera.step == 0          # nothing was aimed at


def test_the_guided_path_looks_around_on_a_worker_thread(mandel, monkeypatch):
    """A 96x54 escape map does not belong in a frame, so the search runs off
    the GUI thread and the result is collected on a later one."""
    monkeypatch.setattr(mandel.module, "plan_guided_step",
                        lambda *_a, **_k: (0.25, -0.125, 9.0))
    # Built on the fixed path and switched afterwards, so the search below
    # is the first one and not a second reading of one the build started.
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    mandel.saved["path"] = "guided"

    canvas._mandelbrot_uniforms(0.0)                 # starts the search
    assert canvas._plan is not None
    assert _until(lambda: canvas._plan["done"])
    span = canvas._plan["target"]
    assert span == (0.25, -0.125)

    canvas._mandelbrot_uniforms(0.0)                 # collects it
    assert canvas._plan is None
    assert canvas._camera.target is not None
    assert canvas._camera.step == 1
    aimed_at = canvas._camera.next_steer
    assert aimed_at > 0.0

    # Not due again yet, so the frame after that only advances the follow.
    canvas._mandelbrot_uniforms(0.0)
    assert canvas._plan is None
    assert canvas._camera.next_steer == aimed_at

    # And a strength of zero means DO NOT STEER: with no reach there is no
    # direction to look in, so every choice would be arbitrary.
    mandel.saved["steering_strength"] = 0.0
    canvas._camera.centre = (0.5, 0.25)
    held = canvas._mandelbrot_uniforms(0.0)
    assert tuple(float(v) for v in held["u_center_offset"]) == (
        pytest.approx(0.5), pytest.approx(0.25))


def test_a_search_that_finds_nothing_asks_again_sooner(mandel, monkeypatch,
                                                       caplog):
    """Rather than giving up on steering for the rest of the dive."""
    def _broken(*_args, **_kwargs):
        raise RuntimeError("the escape map could not be built")

    monkeypatch.setattr(mandel.module, "plan_guided_step", _broken)
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)
    mandel.saved["path"] = "guided"

    with caplog.at_level("DEBUG", logger=F.LOG.name):
        canvas._mandelbrot_uniforms(0.0)
        assert _until(lambda: canvas._plan["done"])
        canvas._mandelbrot_uniforms(0.0)

    assert "could not plan a steering step" in caplog.text
    camera = canvas._camera
    assert camera.target is None
    assert camera.step == 0
    # 0.35 of the interval rather than a whole one.
    assert 0.0 < camera.next_steer < 0.4


def test_the_reference_follows_the_view_down(mandel, monkeypatch, caplog):
    """Each refinement is picked out of the CURRENT view, and the view
    shrinks: refining every decade holds the picture sharp to eleven,
    against one or two without."""
    module = mandel.module
    canvas = mandel.build()
    canvas._orbit = _StandInOrbit(max_iter=8, digits=20)

    canvas._mandelbrot_uniforms(0.0)                 # schedules the first one
    due = canvas._refine_due
    assert due == pytest.approx(canvas._depth + module.REFINE_EVERY, abs=0.01)
    canvas._mandelbrot_uniforms(0.0)                 # not due yet
    assert canvas._refine_due == due

    fresh = _StandInOrbit(max_iter=8, digits=20)
    monkeypatch.setattr(module, "best_reference_in_view",
                        lambda *_a, **_k: (0.1, 0.2))
    monkeypatch.setattr(module, "rebased_orbit",
                        lambda *_a, **_k: ((0.1, 0.2), fresh))

    # A survey already running is not started twice.
    canvas._refine_due = 0.0
    canvas._refine_thread_running = True
    canvas._mandelbrot_uniforms(0.0)
    assert _until(lambda: canvas._refined is not None, seconds=0.3) is False

    canvas._refine_thread_running = False
    canvas._camera.centre = (0.3, 0.4)
    canvas._mandelbrot_uniforms(0.0)                 # starts the survey
    assert _until(lambda: canvas._refined is not None)

    canvas._mandelbrot_uniforms(0.0)                 # lands it
    assert canvas._orbit is fresh
    # The camera is sitting ON the new reference, so its offset starts again.
    assert canvas._camera.centre == (0.0, 0.0)
    assert canvas._refine_due == pytest.approx(
        canvas._depth + module.REFINE_EVERY, abs=0.01)


def test_a_refinement_that_fails_keeps_the_reference_it_has(mandel,
                                                            monkeypatch,
                                                            caplog):
    """A poor reference draws noise, where an old one merely draws a view
    that is off centre."""
    module = mandel.module
    canvas = mandel.build()
    orbit = _StandInOrbit(max_iter=8, digits=20)
    canvas._orbit = orbit

    def _broken(*_args, **_kwargs):
        raise RuntimeError("the escape map could not be built")

    monkeypatch.setattr(module, "best_reference_in_view", _broken)
    canvas._mandelbrot_uniforms(0.0)                 # schedules
    canvas._refine_due = 0.0
    with caplog.at_level("DEBUG", logger=F.LOG.name):
        canvas._mandelbrot_uniforms(0.0)             # starts the survey
        assert _until(lambda: canvas._refined is not None)
        canvas._mandelbrot_uniforms(0.0)             # lands nothing

    assert "could not refine the reference" in caplog.text
    assert canvas._orbit is orbit
    assert canvas._refine_due == pytest.approx(
        canvas._depth + module.REFINE_EVERY, abs=0.01)
