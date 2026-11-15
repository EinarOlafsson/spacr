"""Fallback and error paths of the 2026-10-12 backdrop work.

Each test drives a path the field takes when something it relies on is
missing or gone: a screen that cannot report its refresh rate, a canvas or
density Automatic has no measurement for, the Python grain kernel that runs
before (or instead of) the compiled one, a fold that carries no usable
centre, a shading thread whose repaint doorbell was deleted, and a pointer
move that arrives while the tick clock is not running.
"""
from __future__ import annotations

import math
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "field.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    return settings


def test_a_screen_lookup_that_raises_falls_back_to_sixty_hertz(monkeypatch):
    from spacr.qt import hidpi

    def broken(widget):
        raise RuntimeError("display server went away")

    monkeypatch.setattr(hidpi, "screen_for_widget", broken)
    assert ambient._display_hz(None, ambient.FIELD_FPS) == pytest.approx(60.0)
    assert ambient._display_hz(None, 25) == pytest.approx(20.0)


@pytest.mark.parametrize("theme", ["blobs", "drift", "data_art_fungal_growth"])
def test_automatic_keeps_an_unmeasured_canvas_on_the_cpu(theme):
    engine = ambient.make_engine(theme, "spacr", "#191919", seed=2)
    assert engine._graphics_automatic(0, 480) is False
    engine.density = 0.0
    assert engine._graphics_automatic(3840, 2160) is False


@pytest.mark.parametrize("point", [
    ("rings", "left", 0.5),
    ("rings", None, 0.5),
    ("rings", math.nan, 0.5),
    ("rings", 0.5, math.inf),
])
def test_a_fold_without_a_usable_centre_sends_no_rings(point):
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418", seed=4)
    engine._fold_waves.clear()
    engine._add_ripple(point)
    assert not engine._fold_waves
    engine._add_ripple(("rings", 2.0, -1.0))
    assert len(engine._fold_waves) == ambient.FOLD_RINGS
    assert engine._fold_waves[0][1] == (1.0, 0.0)


def test_the_shading_thread_survives_a_deleted_doorbell():
    import shiboken6

    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                                 seed=9, density=0.1)
    producer = ambient._FrameProducer(engine, ambient.threading.Lock(), 1, (48, 32))
    doorbell = QObject()
    producer.doorbell = doorbell
    shiboken6.delete(doorbell)
    assert not shiboken6.isValid(doorbell)
    producer.start()
    try:
        deadline = time.monotonic() + 60.0
        while producer.frames_shaded < 4 and time.monotonic() < deadline:
            producer._wake.set()
            time.sleep(0.05)
        assert producer.frames_shaded >= 4
        assert producer.is_alive()
        assert producer.latest() is not None
    finally:
        producer.stop()
    assert not producer.is_alive()


def test_a_move_while_the_tick_clock_is_stopped_moves_the_pointer_but_not_time(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(360, 240)
    field = ambient.install_ambient(host, theme="data_art_impulse_lens",
                                    palette="spacr", seed=5, gravity_radius=0.1)
    host.show()
    qtbot.waitExposed(host)
    field._timer.setInterval(100_000)
    assert field._interaction_app is not None
    field._clock.invalidate()
    art = field._art_input
    elapsed, serial = art._elapsed, art._serial
    position = QPoint(120, 80)
    event = QMouseEvent(QEvent.MouseMove, QPointF(position),
                        QPointF(host.mapToGlobal(position)), Qt.NoButton,
                        Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(host.window().windowHandle(), event)
    assert art._serial == serial + 1
    assert art._elapsed == elapsed
    assert art._snapshot[2] == pytest.approx(
        ((120 + .5) / field.width(), (80 + .5) / field.height()), abs=.01)
