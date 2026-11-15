"""The fractal backdrop's CPU half: pointer, thread budget, engine ring and widget lifecycle.

``fractal_travel`` is a renderer wrapped in a great deal of declining to fail:
a pointer that answers even for a freed widget, a thread budget that survives
an unreadable Numba cap, a four-frame antialiasing ring, a render thread
joined however the widget goes, and a CPU widget that keeps its frame loop
honest through pauses, failures and resizes.  The CPU widget is built against
a cheap engine (the builder reads the module-level ``OrbitEngine``), so the
frame loop is real -- a real QThread, real signals, real timers -- while a
frame costs a memset.  One general test per unit, each walking a case table
with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QSize, Qt  # noqa: E402
from PySide6.QtGui import QResizeEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from spacr.qt.widgets import fractal_travel as F  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def _no_mouse_button_is_held(monkeypatch):
    """A held button (left over from another test) suppresses the pull; pin it released."""
    monkeypatch.setattr(QApplication, "mouseButtons",
                        staticmethod(lambda: Qt.MouseButton.NoButton))


class _StubWidget:
    """Just enough widget for ``Pointer.sample``: a fixed mapped position."""

    def __init__(self, point, visible=True, width=200, height=100):
        self._point, self._visible = point, visible
        self._width, self._height = width, height

    def isVisible(self):
        return self._visible

    def mapFromGlobal(self, _global):
        return self._point

    def width(self):
        return self._width

    def height(self):
        return self._height


class _Freed:
    def isVisible(self):
        raise RuntimeError("Internal C++ object (QWidget) already deleted.")


def test_the_pointer_is_outside_whenever_it_cannot_be_placed(qapp, subtests):
    """No widget, a hidden or freed widget, or a position off the widget all read as outside."""
    centre = _StubWidget(QPoint(100, 50))
    for case_id, target in [("no widget", None),
                            ("hidden widget", _StubWidget(QPoint(100, 50), visible=False)),
                            ("freed widget", _Freed())]:
        with subtests.test(msg=case_id):
            pointer = F.Pointer()
            assert pointer.sample(centre).inside is True
            assert pointer.sample(target) is pointer
            assert pointer.inside is False

    with subtests.test(msg="off the widget the pull decays"):
        pointer = F.Pointer()
        for _ in range(40):
            pointer.sample(centre)
        held = pointer.pull
        assert held > 0.5
        pointer.sample(_StubWidget(QPoint(-40, 50)))
        assert pointer.inside is False
        assert pointer.pull == pytest.approx(held - 0.08)
        assert pointer.push == 0.0


class _NumbaCap:
    def __init__(self, value):
        self.NUMBA_NUM_THREADS = value


def test_resolved_cpu_threads_keeps_cores_for_the_application(monkeypatch, subtests):
    """Numba's cap is honoured when readable, small machines keep a core, an explicit count is clamped."""
    cases = [
        ("numba caps at four", _NumbaCap(4), 32, None, 3),
        ("unreadable numba cap", _NumbaCap("as many as you like"), 32, None, 19),
        ("no numba, eight cores", None, 8, None, 6),
        ("no numba, six cores", None, 6, None, 5),
        ("no numba, four cores", None, 4, None, 3),
        ("no numba, two cores", None, 2, None, 1),
        ("explicit count clamped to the machine", None, 8, 64, 8),
        ("explicit count of zero", None, 8, 0, 1),
    ]
    for case_id, numba_config, cpus, explicit, expected in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as m:
            m.setattr(F, "numba_config", numba_config)
            assert F.resolved_cpu_threads(F.Settings(cpu_threads=explicit),
                                          F.HardwareProfile(logical_cpus=cpus)) == expected


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
            raise RuntimeError("Internal C++ object (QThread) already deleted.")

    def wait(self, milliseconds):
        self.waited = milliseconds
        return True


def test_join_on_destroy_joins_quietly_or_not_at_all(monkeypatch, subtests):
    """The destroyed handler quits and waits 5 s, swallows every failure, and is skipped without a signal."""
    with subtests.test(msg="a thread is quit and waited for"):
        widget, thread = _WidgetWithDestroyed(), _Thread()
        F._join_on_destroy(widget, thread)
        widget.destroyed.handler()
        assert (thread.quits, thread.waited) == (1, 5000)

    with subtests.test(msg="a freed thread does not raise"):
        widget, thread = _WidgetWithDestroyed(), _Thread(refuses=True)
        F._join_on_destroy(widget, thread)
        widget.destroyed.handler()
        assert (thread.quits, thread.waited) == (1, None)

    with subtests.test(msg="a join helper that raises is swallowed"), monkeypatch.context() as m:
        m.setattr(F, "_quit_and_join_thread",
                  lambda _t: (_ for _ in ()).throw(RuntimeError("already gone")))
        widget = _WidgetWithDestroyed()
        F._join_on_destroy(widget, object())
        assert widget.destroyed.handler is not None
        widget.destroyed.handler()

    with subtests.test(msg="no destroyed signal to connect"):
        widget = _WidgetWithDestroyed(refuses=True)
        F._join_on_destroy(widget, _Thread())
        assert widget.destroyed.handler is None


def test_orbit_engine_keeps_a_four_frame_jitter_ring(subtests):
    """Frames have the asked size, the first fills the ring, each advances a phase, and a resize restarts it."""
    pytest.importorskip("numba")

    with subtests.test(msg="first frame"):
        engine = F.OrbitEngine(1)
        assert (engine.ring, engine.output, engine.frames) == (None, None, 0)
        first = engine.render(12, 10, 0.0, 1.0, 0.5, 6)
        assert (first.shape, first.dtype) == ((10, 12, 3), np.dtype(np.uint8))
        assert engine.frames == 1
        assert int(first.max()) > 0
        for index in range(4):
            np.testing.assert_array_equal(engine.ring[index], engine.ring[0])

    with subtests.test(msg="one jitter phase per frame"):
        assert len(F.JITTERS) == 4
        engine = F.OrbitEngine(1)
        for step in range(5):
            engine.render(12, 10, 0.05 * step, 1.0, 0.5, 6)
            assert engine.slot == (step + 1) % 4
        assert engine.frames == 5

    with subtests.test(msg="returned frame is a copy"):
        engine = F.OrbitEngine(1)
        first = engine.render(12, 10, 0.0, 1.0, 0.5, 6)
        keep = first.copy()
        engine.render(12, 10, 9.0, 1.0, 0.5, 6)
        np.testing.assert_array_equal(first, keep)

    with subtests.test(msg="same size keeps buffers, new size restarts"):
        engine = F.OrbitEngine(1)
        engine.render(12, 10, 0.0, 1.0, 0.5, 6)
        ring, output = engine.ring, engine.output
        engine.render(12, 10, 0.1, 1.0, 0.5, 6)
        assert engine.ring is ring and engine.output is output
        assert engine.frames == 2
        frame = engine.render(20, 14, 0.2, 1.0, 0.5, 6)
        assert frame.shape == (14, 20, 3)
        assert engine.ring.shape == (4, 14, 20, 3)
        assert (engine.frames, engine.slot) == (1, 1)


class _CheapEngine:
    """Answers a frame instantly and records what it was asked."""

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
def make_cpu_widget(qapp, monkeypatch):
    """Build live CPU widgets with a cheap engine and shut them all down afterwards."""
    pytest.importorskip("numba")
    monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)
    built = []

    def make():
        widget = F._make_cpu_widget(F.Settings(pattern="orbit", backend="cpu"),
                                    F.RuntimeControls(),
                                    F.HardwareProfile(logical_cpus=4))
        built.append(widget)
        return widget

    yield make
    for widget in built:
        widget.shutdown()
        widget.deleteLater()


def _frame(width=8, height=6, value=7):
    return np.full((height, width, 3), value, dtype=np.uint8)


def _timing(widget):
    return widget.stats_text().splitlines()[-1]


def test_the_cpu_widget_frame_loop_delivers_adapts_and_reports(make_cpu_widget, subtests):
    """Requests reach the engine, failures become text, late frames are dropped, scale tracks frame time."""
    with subtests.test(msg="the worker unpacks the request"):
        widget = make_cpu_widget()
        delivered = []
        widget._worker.frame_ready.connect(lambda frame, seconds: delivered.append((frame, seconds)),
                                           Qt.ConnectionType.DirectConnection)
        widget._worker.render({"width": 8, "height": 6, "t": 2.5, "speed": 3.0,
                               "dream": 1.25, "iterations": 5})
        assert widget._worker.engine.calls[-1] == dict(
            width=8, height=6, t=2.5, speed=3.0, dream=1.25, iterations=5,
            pointer_x=0.0, pointer_y=0.0, pull=0.0, push=0.0)
        frame, seconds = delivered[-1]
        np.testing.assert_array_equal(frame, _frame(8, 6))
        assert seconds >= 0.0

    with subtests.test(msg="a frame that cannot be shaded is reported"):
        widget = make_cpu_widget()
        complaints = []
        widget._worker.failed.connect(complaints.append, Qt.ConnectionType.DirectConnection)
        widget._worker.engine.explode = ValueError("no memory for that frame")
        widget._worker.render({"width": 8, "height": 6, "t": 0.0, "speed": 3.0,
                               "dream": 1.0, "iterations": 5})
        assert complaints == ["ValueError: no memory for that frame"]
        assert widget._error == "ValueError: no memory for that frame"

    with subtests.test(msg="a paused widget asks for no more frames"):
        widget = make_cpu_widget()
        requests = []
        widget.render_requested.connect(requests.append)
        widget.show()
        try:
            widget._request_frame()
            assert len(requests) == 1 and requests[0]["iterations"] == 5
            widget.pause()
            widget._request_frame()
            assert len(requests) == 1
        finally:
            widget.hide()

    with subtests.test(msg="a frame after shutdown is dropped"):
        widget = make_cpu_widget()
        widget._accept_frame(_frame(), 0.01)
        shown = widget._image_array
        assert widget._image.width() == 8
        widget.shutdown()
        widget._accept_frame(_frame(4, 4, value=200), 0.02)
        assert widget._image_array is shown

    with subtests.test(msg="paused shows the last frame and schedules none"):
        widget = make_cpu_widget()
        widget._timer.stop()
        widget._accept_frame(_frame(8, 6), 0.001)
        assert widget._timer.isActive() is True
        widget.pause()
        assert widget._timer.isActive() is False
        widget._accept_frame(_frame(12, 10), 0.001)
        assert widget._image.width() == 12
        assert widget._timer.isActive() is False

    with subtests.test(msg="the render scale follows the frame time"):
        widget = make_cpu_widget()

        def settle(seconds):
            widget._frames = 23
            widget._render_ema = seconds
            widget._accept_frame(_frame(), seconds)
            return widget._adaptive_scale

        assert widget._adaptive_scale == 1.0
        slow = settle(0.1)
        assert slow == pytest.approx(0.82)
        assert settle(0.020) == slow
        assert settle(0.002) > slow

    with subtests.test(msg="a failure is shown and retried unless paused"):
        widget = make_cpu_widget()
        widget._timer.stop()
        widget._busy = True
        widget._on_failure("ValueError: no memory for that frame")
        assert _timing(widget) == "ValueError: no memory for that frame"
        assert (widget._busy, widget._timer.isActive()) == (False, True)
        widget.pause()
        widget._on_failure("ValueError: and again")
        assert _timing(widget) == "ValueError: and again"
        assert widget._timer.isActive() is False

    with subtests.test(msg="the stats line"):
        widget = make_cpu_widget()
        assert widget.stats_text().startswith(f"v{F.VERSION} · CPU/")
        assert _timing(widget) == "compiling"
        widget._accept_frame(_frame(), 0.0125)
        assert _timing(widget) == "12.5 ms frame"
        widget.pause()
        assert _timing(widget) == "paused for a run"

    with subtests.test(msg="a resize asks for a fresh frame unless paused"):
        widget = make_cpu_widget()

        def resize_to(width, height):
            old = QSize(widget.width(), widget.height())
            widget.resize(width, height)
            QApplication.sendEvent(widget, QResizeEvent(QSize(width, height), old))

        widget._timer.stop()
        resize_to(700, 500)
        assert widget._timer.isActive() is True
        widget.pause()
        resize_to(640, 480)
        assert widget._timer.isActive() is False


def test_the_cpu_widget_pauses_sizes_and_shuts_down_for_good(make_cpu_widget, qapp, monkeypatch, subtests):
    """Pause/resume report change, the render size falls back sanely, and every shutdown joins the thread."""
    with subtests.test(msg="without numba the backend refuses at build"), monkeypatch.context() as m:
        m.setattr(F, "njit", None)
        with pytest.raises(RuntimeError, match="numba"):
            F._make_cpu_widget(F.Settings(backend="cpu"), F.RuntimeControls(),
                               F.HardwareProfile(logical_cpus=4))

    with subtests.test(msg="pausing and resuming twice"):
        widget = make_cpu_widget()
        assert [widget.pause(), widget.pause(), widget.is_paused()] == [True, False, True]
        assert [widget.resume(), widget.resume(), widget.is_paused()] == [True, False, False]
        assert [widget.set_animating(False), widget.set_animating(False)] == [True, False]

    with subtests.test(msg="a stopped widget does not resume"):
        widget = make_cpu_widget()
        widget.pause()
        widget.shutdown()
        assert (widget.resume(), widget.is_paused()) == (False, True)

    with subtests.test(msg="render scale unreadable or zero"), monkeypatch.context() as m:
        widget = make_cpu_widget()
        widget.resize(1600, 1200)
        m.setattr(F, "_render_scale", lambda: 1.0)
        native = widget._target_size()

        def _broken():
            raise RuntimeError("the settings store is not there")

        m.setattr(F, "_render_scale", _broken)
        assert widget._target_size() == native
        m.setattr(F, "_render_scale", lambda: 0.0)
        budgeted = widget._target_size()
        assert budgeted[0] < native[0] and budgeted[1] < native[1]
        assert budgeted[0] * budgeted[1] == pytest.approx(460_000, rel=0.05)

    for case_id, prepare, act in [
            ("shutdown", lambda w: None, lambda w: w.shutdown()),
            ("shutdown twice", lambda w: None, lambda w: (w.shutdown(), w.shutdown())),
            ("a cleared quit hook", lambda w: setattr(w, "_app_quit_join", None),
             lambda w: w.shutdown()),
            ("closing the widget", lambda w: None, lambda w: w.close())]:
        with subtests.test(msg=case_id):
            widget = make_cpu_widget()
            assert widget._stopped is False
            prepare(widget)
            act(widget)
            assert widget._stopped is True
            assert widget._timer.isActive() is False
            assert widget._thread.isRunning() is False

    with subtests.test(msg="a backdrop freed with its screen takes its quit hook"):
        import shiboken6
        from PySide6.QtCore import SIGNAL, QEvent
        from PySide6.QtWidgets import QWidget

        before = qapp.receivers(SIGNAL("aboutToQuit()"))
        for _ in range(3):
            was = qapp.receivers(SIGNAL("aboutToQuit()"))
            screen = QWidget()
            backdrop = F._make_cpu_widget(F.Settings(pattern="orbit", backend="cpu"),
                                          F.RuntimeControls(), F.HardwareProfile(logical_cpus=4))
            backdrop.setParent(screen)
            assert qapp.receivers(SIGNAL("aboutToQuit()")) == was + 1
            screen.deleteLater()
            qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            qapp.processEvents()
            assert not shiboken6.isValid(backdrop)
        assert qapp.receivers(SIGNAL("aboutToQuit()")) == before


def test_create_fractal_widget_registers_by_identity_and_falls_back_honestly(qapp, qtbot, monkeypatch, caplog, subtests):
    """Controls register once per object, a GPU-only pattern falls back to the orbit fold and says so."""
    pytest.importorskip("numba")
    monkeypatch.setattr(F, "_LIVE_CONTROLS", [])
    monkeypatch.setattr(F, "OrbitEngine", _CheapEngine)

    with subtests.test(msg="registered by identity, not value"):
        settings = F.Settings(pattern="orbit", backend="cpu")
        controls, twin = F.RuntimeControls(), F.RuntimeControls()
        assert twin == controls and twin is not controls
        built = [F.create_fractal_widget(settings, controls),
                 F.create_fractal_widget(settings, controls),
                 F.create_fractal_widget(settings, twin)]
        try:
            assert sum(1 for c in F._LIVE_CONTROLS if c is controls) == 1
            assert sum(1 for c in F._LIVE_CONTROLS if c is twin) == 1
        finally:
            for widget in built:
                widget.shutdown()
                widget.deleteLater()

    with subtests.test(msg="mandelbrot without a GPU becomes the orbit fold"), monkeypatch.context() as m:
        m.setattr(F, "platform_can_do_opengl", lambda: True)
        m.setattr(F, "gpu_is_available", lambda: True)

        def refuse(*_args, **_kwargs):
            raise F.GpuBackendError("the fractal shaders do not compile here")

        m.setattr(F, "_make_gpu_widget", refuse)
        caplog.clear()
        with caplog.at_level("WARNING", logger=F.LOG.name):
            widget = F.create_fractal_widget(F.Settings(pattern="mandelbrot", backend="gpu"))
        try:
            assert widget.backend_name == "cpu"
            assert f" · {F.FALLBACK_PATTERN} · " in widget.stats_text()
            assert "mandelbrot" not in widget.stats_text()
            assert "the fractal shaders do not compile here" in caplog.text
            assert "needs the GPU renderer" in caplog.text
        finally:
            widget.shutdown()
            widget.deleteLater()

    # The ``except (RuntimeError, TypeError)`` around shutdown's disconnect is
    # not reachable on PySide6: a slot removed first only warns (asserted
    # below), and an application whose disconnect raises needs a global
    # ``QApplication.instance`` patch that breaks qtbot.  What matters -- the
    # thread is joined however the disconnect went -- is asserted instead.
    for case_id, disconnect_first in [("an ordinary shutdown", False),
                                      ("a quit slot already disconnected", True)]:
        with subtests.test(msg=case_id):
            widget = F.create_fractal_widget(F.Settings(backend="cpu"))
            qtbot.addWidget(widget)
            if disconnect_first:
                QApplication.instance().aboutToQuit.disconnect(widget._app_quit_join)
                with pytest.warns(RuntimeWarning, match="Failed to disconnect"):
                    widget.shutdown()       # PySide6 warns rather than raising
            else:
                widget.shutdown()
            assert widget._stopped is True
            assert widget._thread.isRunning() is False
            widget.shutdown()
            assert widget._stopped is True
