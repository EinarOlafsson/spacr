"""The backdrop's refusals: no screen, no canvas, no thread, no host.

This widget paints behind every module screen while a pipeline runs, so the
rule the whole module is written to is that a missing answer costs a frame and
never the screen. The paths below are the ones that only a real machine
reaches -- a headless plugin that reports no screen, a fractal bud that lands
off the buffer, a shading thread that was never started, a paint that arrives
while the shader holds the engine, and a fractal backdrop being torn down
underneath the code that is tidying it away.

Each test drives the refusal next to the case that does the work, because
"nothing happened" is only a claim about the guard if the same test shows
what happening looks like.
"""

from __future__ import annotations

import threading

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent                               # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter              # noqa: E402
from PySide6.QtWidgets import QWidget                           # noqa: E402

from spacr.qt.widgets import ambient as amb                     # noqa: E402
from spacr.qt.widgets.ambient import (BUFFER_MAX_PIXELS,        # noqa: E402
                                      BUFFER_MIN_EDGE,
                                      AmbientWidget, Form,
                                      _FractalTracksItsHost,
                                      _FrameProducer,
                                      _retire_fractals_on,
                                      _retire_producer,
                                      _the_spaceout_fractal,
                                      make_engine, screen_pixels)

pytestmark = pytest.mark.qt

DARK = "#101418"


def _fractal():
    return make_engine("fractal", "rainbow", DARK, seed=7)


def _numpy():
    return pytest.importorskip("numpy")


def _paint(engine, width=320, height=200) -> QImage:
    image = QImage(width, height, QImage.Format_RGB32)
    painter = QPainter(image)
    painter.fillRect(image.rect(), QColor(DARK))
    engine.paint(painter, width, height)
    painter.end()
    return image


# ---------------------------------------------------------------------------
# Which screen the buffer is sized for
# ---------------------------------------------------------------------------

def test_a_machine_that_reports_no_screen_gets_the_fallback_ceiling(
        qapp, monkeypatch):
    """A ceiling is required, so a missing screen answers with the default.

    Returning 0 or raising here would size the render buffer at nothing and
    the backdrop would go black on exactly the machines -- headless plugins,
    a widget built before its window exists -- that this ceiling exists for.
    """
    real = screen_pixels()
    assert real > 0 and real != BUFFER_MAX_PIXELS, (
        "this session has no real screen to contrast the fallback with")

    class _NoScreens:
        @staticmethod
        def primaryScreen():
            return None

    monkeypatch.setattr("PySide6.QtGui.QGuiApplication", _NoScreens)
    assert screen_pixels() == BUFFER_MAX_PIXELS


def test_a_widget_that_raises_when_asked_its_screen_gets_the_fallback(qapp):
    """A widget mid-teardown raises rather than answering None."""
    asked = []

    class _Gone:
        def screen(self):
            asked.append("screen")
            raise RuntimeError("this widget has gone")

    assert screen_pixels(_Gone()) == BUFFER_MAX_PIXELS
    assert asked == ["screen"], "the widget was never asked"


def test_setting_the_ceiling_it_already_has_does_not_drop_the_buffer(qapp):
    """A window moved between two identical screens must not restart the
    field: dropping the buffer is a reallocation and a visible reshade."""
    engine = _fractal()
    _paint(engine)
    assert engine._buffer is not None, "nothing was buffered to begin with"

    engine.set_max_pixels(engine.max_pixels)
    assert engine._buffer is not None, "an unchanged ceiling dropped the buffer"

    smaller = BUFFER_MIN_EDGE ** 2
    engine.set_max_pixels(smaller)
    assert engine.max_pixels == smaller
    assert engine._buffer is None


# ---------------------------------------------------------------------------
# The fractal engine's cost guard
# ---------------------------------------------------------------------------

def test_a_frame_with_nothing_measured_leaves_the_capacity_guard_alone(qapp):
    """The guard adapts to measured costs, and no measurement is no news.

    Moving it on an empty batch would walk the render capacity around on the
    strength of whatever `dt` happened to be, which is the one input that
    says nothing about how fast this machine is.
    """
    engine = _fractal()
    start = engine.afford()

    engine.advance(0.1)
    assert engine.afford() == start, "an unmeasured frame moved the guard"

    engine._spent = [1000.0]        # a pass far over the frame budget
    engine.advance(0.1)
    assert engine.afford() < start


def test_the_measured_costs_are_a_window_and_not_a_log(qapp):
    """`_spent` is appended to on every shading pass and drained on every
    advance, and a widget that is painted but never advanced would otherwise
    grow it without bound for as long as the screen is open."""
    engine = _fractal()
    engine._spent = [0.0] * 10
    _paint(engine)
    assert len(engine._spent) == 11, "the pass was not measured at all"

    engine._spent = [0.0] * 20
    _paint(engine)
    assert len(engine._spent) == 16
    assert engine._spent[-1] > 0.0, "the newest measurement was trimmed away"


# ---------------------------------------------------------------------------
# The fractal engine's buds
# ---------------------------------------------------------------------------

def test_a_canvas_with_no_pixels_grows_no_buds(qapp):
    """A widget shown at zero size still ticks, and half a bud geometry
    divided by a zero span is a NaN that reaches the shader."""
    engine = _fractal()
    assert engine.buds(0, 200) == ()
    assert engine.buds(320, 0) == ()
    assert len(engine.buds(320, 200)) >= 1


def test_a_bud_that_falls_off_the_buffer_impresses_nothing(qapp):
    """A bud whose window clips to a strip outside its own rim.

    The blend share is zero across the whole strip, and going on would write
    the bud's sampling into the field at weight zero -- the same picture, for
    the cost of the arithmetic.
    """
    np = _numpy()
    engine = _fractal()
    bud = engine.buds(320, 200)[0]
    reach = bud.radius * (1.0 + amb.FRACTAL_BUD_FEATHER)

    off = bud._replace(cx=-reach + 0.1, cy=5.0)
    zr = np.full((200, 320), 1.0, dtype=np.float32)
    zi = np.full((200, 320), 1.0, dtype=np.float32)
    assert engine._impress(off, zr, zi, None, 320, 200) is None
    assert np.all(zr == np.float32(1.0)), "a bud off the buffer moved the field"

    zr = np.full((200, 320), 1.0, dtype=np.float32)
    zi = np.full((200, 320), 1.0, dtype=np.float32)
    engine._impress(bud, zr, zi, None, 320, 200)
    assert not np.all(zr == np.float32(1.0))


def test_a_bud_with_no_tunnel_fades_the_bands_it_lands_on(qapp):
    """The rim is a blend, so a bud with no rings of its own does not paste
    the main form's rings across itself -- it dissolves them over its own
    radius, which is what makes the edge refract rather than stick."""
    np = _numpy()
    engine = _fractal()
    plain = engine.buds(320, 200)[0]._replace(tunnel=0.0)

    bands = np.full((200, 320), 4.0, dtype=np.float32)
    zr = np.full((200, 320), 1.0, dtype=np.float32)
    zi = np.full((200, 320), 1.0, dtype=np.float32)
    same = engine._impress(plain, zr, zi, bands, 320, 200)
    assert same is bands, "a new band buffer was allocated for a bud with none"
    assert float(bands.min()) == pytest.approx(0.0, abs=1e-5), (
        "the bud's centre kept the main form's rings")
    assert float(bands.max()) == pytest.approx(4.0), (
        "the rings outside the bud were faded too")

    # With no bands anywhere there is nothing to fade and nothing to make.
    zr = np.full((200, 320), 1.0, dtype=np.float32)
    zi = np.full((200, 320), 1.0, dtype=np.float32)
    assert engine._impress(plain, zr, zi, None, 320, 200) is None


# ---------------------------------------------------------------------------
# The shading thread
# ---------------------------------------------------------------------------

def test_a_producer_that_was_never_started_is_stopped_without_a_join(qapp):
    """`stop` is called on every hide, including the ones before a show."""
    engine = _fractal()
    producer = _FrameProducer(engine, threading.RLock(), 30, (64, 64))
    assert producer.is_alive() is False

    producer.stop()
    assert producer._stop.is_set()
    assert producer.is_alive() is False

    running = _FrameProducer(engine, threading.RLock(), 30, (64, 64))
    running.start()
    running.stop()
    assert running.is_alive() is False, "a started thread was not joined"


def test_an_already_emptied_box_retires_nothing_twice(qapp):
    """The box is emptied before the producer is stopped, so the widget's
    `destroyed` slot cannot stop the same thread a second time."""
    stopped = []

    class _Producer:
        def stop(self):
            stopped.append("stop")

    box = [_Producer()]
    _retire_producer(box)
    assert box == [None] and stopped == ["stop"]

    _retire_producer(box)
    assert stopped == ["stop"], "an emptied box stopped something"

    _retire_producer([])
    assert stopped == ["stop"]


# ---------------------------------------------------------------------------
# The widget's frames
# ---------------------------------------------------------------------------

def test_a_republish_with_no_canvas_publishes_nothing(qtbot):
    """A live Preferences change lands while the widget is at zero size.

    The engine has no frame to give for an empty canvas, and publishing the
    None would hand the paint path a slot it has to test for anyway.
    """
    widget = AmbientWidget(theme="blobs", palette="spacr", background=DARK,
                           seed=3)
    qtbot.addWidget(widget)
    widget.resize(240, 160)
    producer = _FrameProducer(widget.engine, widget._engine_lock, 30, (0, 0))
    widget._producer_box[0] = producer

    widget.set_time(3.0)
    assert producer.latest() is None, "an empty canvas published a frame"

    producer.size = (240, 160)
    widget.set_time(4.0)
    assert producer.latest() is not None


def test_a_first_frame_that_cannot_be_shaded_is_not_published(qtbot,
                                                              monkeypatch):
    """The producer is installed either way; only the head start is lost."""
    monkeypatch.setattr(_FrameProducer, "start", lambda self: None)

    widget = AmbientWidget(theme="blobs", palette="spacr", background=DARK,
                           seed=3)
    qtbot.addWidget(widget)
    widget.resize(240, 160)
    monkeypatch.setattr(widget.engine, "shade", lambda width, height: None)

    widget._start_producer()
    producer = widget._producer_box[0]
    assert producer is not None
    assert producer.latest() is None

    other = AmbientWidget(theme="blobs", palette="spacr", background=DARK,
                          seed=3)
    qtbot.addWidget(other)
    other.resize(240, 160)
    other._start_producer()
    assert other._producer_box[0].latest() is not None, (
        "the ordinary path publishes its first frame here")


def test_a_screen_change_the_engine_refuses_is_not_fatal(qtbot, monkeypatch):
    """A backdrop that cannot work out its ceiling paints at the old one."""
    widget = AmbientWidget(theme="blobs", palette="spacr", background=DARK,
                           seed=3)
    qtbot.addWidget(widget)
    widget.resize(240, 160)

    widget.engine.max_pixels = 123456
    refused = []

    def _refuse(_pixels):
        refused.append("set_max_pixels")
        raise RuntimeError("the engine has gone")

    monkeypatch.setattr(widget.engine, "set_max_pixels", _refuse)
    widget._follow_screen()
    assert refused == ["set_max_pixels"]
    assert widget.engine.max_pixels == 123456

    monkeypatch.undo()
    widget.engine.max_pixels = 123456
    widget._follow_screen()
    assert widget.engine.max_pixels == screen_pixels(widget)


def test_a_paint_that_cannot_take_the_engine_lock_leaves_the_page_flat(
        qtbot):
    """The one promise of this paint path: it never waits for anything.

    With no frame published yet, the synchronous shade is a nicety. If the
    shading thread is mid-frame the paint gives it up and shows the flat page
    rather than blocking the GUI thread on a pass it does not need.
    """
    widget = AmbientWidget(theme="blobs", palette="spacr", background=DARK,
                           seed=3)
    qtbot.addWidget(widget)
    widget.resize(240, 160)
    widget._producer_box[0] = _FrameProducer(
        widget.engine, widget._engine_lock, 30, (240, 160))
    widget._last_frame = None

    painted = []
    widget.engine.paint = lambda *args: painted.append("paint")

    held = threading.Event()
    release = threading.Event()

    def _shade_a_frame():
        with widget._engine_lock:
            held.set()
            release.wait(5.0)

    holder = threading.Thread(target=_shade_a_frame, daemon=True)
    holder.start()
    try:
        assert held.wait(5.0)
        before = widget.repeated_frames
        widget.grab()
        assert widget.repeated_frames > before, "nothing was repainted"
        assert painted == [], "the paint waited for the shading thread"
    finally:
        release.set()
        holder.join(5.0)

    widget._last_frame = None
    widget.grab()
    assert painted == ["paint"], "an idle engine was not painted"


# ---------------------------------------------------------------------------
# Retiring a spaceout fractal
# ---------------------------------------------------------------------------

def test_a_fractal_that_will_not_shut_down_is_still_unparented(qtbot):
    """A backdrop whose C++ half has gone still has to leave the host.

    Leaving it parented is what put four live canvases and four render
    threads on one screen, so neither refusal may stop the removal or the
    count the caller acts on.
    """
    host = QWidget()
    qtbot.addWidget(host)

    class _Stubborn(QWidget):
        backend_name = "vispy"

        def __init__(self, parent):
            super().__init__(parent)
            self.asked = []

        def shutdown(self):
            self.asked.append("shutdown")
            raise RuntimeError("this canvas has gone")

        def setParent(self, parent):
            self.asked.append("setParent")
            raise RuntimeError("this canvas has gone")

    stubborn = _Stubborn(host)
    assert _retire_fractals_on(host) == 1
    assert stubborn.asked == ["shutdown", "setParent"]

    class _Willing(QWidget):
        backend_name = "vispy"

    willing = _Willing(host)
    assert _retire_fractals_on(host) == 2, "the old one was not found again"
    assert willing.parent() is None


def test_a_host_that_cannot_be_searched_retires_nothing(qtbot):
    """`install_ambient` runs against a screen being rebuilt, and a host
    whose children cannot be listed must not stop the new backdrop."""
    asked = []

    class _Gone:
        def findChildren(self, kind):
            asked.append(kind)
            raise RuntimeError("this host has gone")

    assert _retire_fractals_on(_Gone()) == 0
    assert asked == [QWidget], "the host was never searched"


def test_a_theme_that_cannot_be_read_is_not_a_spaceout_launch(qtbot,
                                                              monkeypatch):
    """An unreadable preference is an ordinary launch, not a failed one."""
    looked = []
    monkeypatch.setattr(amb, "_retire_fractals_on",
                        lambda host: looked.append(host) or 0)

    host = QWidget()
    qtbot.addWidget(host)

    def _refuse():
        raise RuntimeError("the settings file is unreadable")

    monkeypatch.setattr("spacr.qt.theme.spaceout_enabled", _refuse)
    assert _the_spaceout_fractal(host) is None
    assert looked == [], "an unreadable theme still tore the host down"

    monkeypatch.setattr("spacr.qt.theme.spaceout_enabled", lambda: True)
    monkeypatch.setattr(amb, "get_fractal_settings", None, raising=False)
    _the_spaceout_fractal(host)
    assert looked == [host], "a spaceout launch did not clear the host first"


def test_the_backdrop_follows_its_host_and_survives_one_that_has_gone(qtbot):
    """The tracker is what keeps the fractal the size of the screen it is
    behind, and it is connected to a host that may outlive the widget."""
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(400, 300)
    backdrop = QWidget(host)
    backdrop.resize(10, 10)

    tracker = _FractalTracksItsHost(backdrop, host)
    assert tracker.eventFilter(host, QEvent(QEvent.Type.Resize)) is False
    assert backdrop.geometry() == host.rect()

    refused = []

    class _Gone:
        def setGeometry(self, _rect):
            refused.append("setGeometry")
            raise RuntimeError("this widget has gone")

    tracker._widget = _Gone()
    assert tracker.eventFilter(host, QEvent(QEvent.Type.Resize)) is False
    assert refused == ["setGeometry"]
