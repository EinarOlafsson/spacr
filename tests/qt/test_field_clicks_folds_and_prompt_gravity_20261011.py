"""Field feedback the maintainer could not get, 2026-10-11.

* "I should be able to create a ripple with a mouse click, which I cannot
  now." A click counted only when every widget between it and the backdrop's
  host was exactly ``QWidget`` or ``QFrame`` with no focus. Every real screen
  is a Python subclass (``AppScreen``, ``Card``, ``Section``), so a probe of
  460 points over Make Masks found 0 that rippled. A click now ripples unless
  it lands on a control or on a widget that handles clicks itself.
* Folding a section closed sends several expanding rings that gather the
  field's dots along each front, instead of one faint column wave.
* Mouse gravity follows mouse events directly and wakes the shading thread,
  which rings the widget to repaint as soon as the frame is ready, instead of
  waiting for the next timer tick, the next shading beat and the next paint.
"""
from __future__ import annotations

import math
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (QApplication, QFrame, QLabel, QLineEdit,
                               QPushButton, QVBoxLayout, QWidget)

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient

pytestmark = pytest.mark.qt


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    settings = QSettings(str(tmp_path / "field.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    return settings


class Screen(QWidget):
    pass


class Card(QFrame):
    pass


class Canvas(QWidget):
    def mousePressEvent(self, event):
        event.accept()


def _field(qtbot, theme="data_art_impulse_lens", **kwargs):
    host = Screen()
    qtbot.addWidget(host)
    host.resize(360, 240)
    options = dict(theme=theme, palette="spacr", seed=5, gravity_radius=0)
    options.update(kwargs)
    field = ambient.install_ambient(host, **options)
    host.show()
    qtbot.waitExposed(host)
    field._timer.setInterval(100_000)
    return host, field


def _waves(field):
    return list(field.engine._popup_waves)


@pytest.mark.parametrize("make", [
    lambda host: Card(host),
    lambda host: QLabel("selectable description", host),
    lambda host: QFrame(Card(host)),
])
def test_a_click_on_a_python_screen_card_or_label_ripples(qtbot, make):
    host, field = _field(qtbot)
    surface = make(host)
    if isinstance(surface, QLabel):
        surface.setTextInteractionFlags(Qt.TextSelectableByMouse)
    top = surface if surface.parentWidget() is host else surface.parentWidget()
    top.setGeometry(40, 40, 200, 120)
    surface.setGeometry(0, 0, 200, 120)
    top.show()
    surface.show()
    qtbot.mouseClick(surface, Qt.LeftButton, pos=QPoint(20, 20))
    qtbot.waitUntil(lambda: bool(_waves(field)), timeout=2000)
    origin = _waves(field)[-1][1]
    expected = field.mapFromGlobal(surface.mapToGlobal(QPoint(20, 20)))
    assert origin == pytest.approx(((expected.x() + .5) / field.width(),
                                    (expected.y() + .5) / field.height()), abs=.01)


@pytest.mark.parametrize("make", [
    lambda host: QPushButton("Run", host),
    lambda host: QLineEdit(host),
    lambda host: Canvas(host),
])
def test_controls_and_canvases_keep_their_clicks_to_themselves(qtbot, make):
    host, field = _field(qtbot)
    control = make(host)
    control.setGeometry(40, 40, 120, 60)
    control.show()
    qtbot.mouseClick(control, Qt.LeftButton, pos=QPoint(10, 10))
    qtbot.wait(ambient.CLICK_RIPPLE_SETTLE_MS * 3)
    assert not _waves(field)


def test_a_press_on_a_control_cancels_an_unreleased_background_click(qtbot):
    host, field = _field(qtbot)
    button = QPushButton("Run", host)
    button.setGeometry(10, 10, 80, 30)
    button.show()
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(200, 150))
    assert field._click_candidate is not None
    qtbot.mousePress(button, Qt.LeftButton, pos=QPoint(5, 5))
    assert field._click_candidate is None
    qtbot.mouseRelease(button, Qt.LeftButton, pos=QPoint(5, 5))
    qtbot.wait(ambient.CLICK_RIPPLE_SETTLE_MS * 3)
    assert not _waves(field)


def test_spaceout_field_theme_also_ripples_on_click(qtbot):
    host, field = _field(qtbot)
    field.set_theme(ambient.DEFAULT_SPACEOUT_THEME)
    if field.theme() not in ambient._FIELD_THEMES:
        pytest.skip("the spaceout field is not paintable here")
    field._sync_run_state()
    qtbot.mouseClick(host, Qt.LeftButton, pos=QPoint(100, 100))
    qtbot.waitUntil(lambda: bool(_waves(field)), timeout=2000)


def test_folding_a_section_sends_several_rings_that_gather_the_dots(qtbot):
    from spacr.qt.widgets.section import Section

    host, field = _field(qtbot)
    layout = QVBoxLayout(host)
    section = Section("Advanced", parent=host)
    layout.addWidget(section)
    section.set_expanded(True)
    qtbot.wait(20)
    field.engine._fold_waves.clear()
    field.engine._column_waves.clear()
    section.set_expanded(False)
    qtbot.waitUntil(lambda: bool(field.engine._fold_waves), timeout=2000)
    rings = field.engine._fold_waves
    assert len(rings) == ambient.FOLD_RINGS
    starts = sorted(started for started, _origin in rings)
    assert starts[1] - starts[0] == pytest.approx(ambient.FOLD_RING_GAP_S)
    centre = field.mapFromGlobal(section.mapToGlobal(section.rect().center()))
    assert rings[0][1] == pytest.approx(((centre.x() + .5) / field.width(),
                                         (centre.y() + .5) / field.height()),
                                        abs=.01)
    assert not field.engine._column_waves


@pytest.mark.parametrize("dark", [True, False])
def test_fold_rings_gather_dots_toward_each_front(dark):
    background = "#191919" if dark else "#f4f4f4"
    width = height = 480

    def frame(rings):
        engine = ambient.make_engine("data_art_impulse_lens", "spacr",
                                     background, seed=4, density=1.0)
        if rings:
            engine._add_ripple(("rings", .3, .3))
        started = engine.time
        engine.advance(0.6)
        captured = {}
        real = engine._point_material

        def keep(w, h, x, y, light, spread=False):
            captured["xy"] = (np.asarray(x) / w, np.asarray(y) / h)
            captured["light"] = np.asarray(light).copy()
            return real(w, h, x, y, light, spread)

        engine._point_material = keep
        engine._frame_impulse_lens(width, height)
        lattice = next(value for key, value in engine._material_cache.items()
                       if key[0] == "impulse_lens")
        return captured, lattice[0], lattice[1], engine.time - started

    plain, lx, ly, age = frame(False)
    ringed, _x, _y, _age = frame(True)
    ex, ey = lx - .3, ly - .3
    distance = np.hypot(ex, ey) + 1e-9
    radial = ((ringed["xy"][0] - plain["xy"][0]) * ex
              + (ringed["xy"][1] - plain["xy"][1]) * ey) / distance
    front = distance - age * ambient.FOLD_SPEED
    width_ = ambient.FOLD_PACKET_WIDTH
    outside = (front > .2 * width_) & (front < width_)
    assert np.mean(radial[outside] < 0) > .9
    assert np.abs(radial).max() * width > 6
    near = np.abs(front) < .3 * width_
    assert ringed["light"][near].mean() > plain["light"][near].mean()


def test_fold_rings_respect_the_ripple_slider(qtbot):
    host, field = _field(qtbot)
    field._set_ripple_intensity(0)
    ambient._publish_widget_ripple(host, None, None, 1.0, vertical=False,
                                   rings=True)
    assert not field.engine._fold_waves
    field.engine.set_ripples_enabled(False)
    field.engine._add_ripple(("rings", .5, .5))
    assert not field.engine._fold_waves


def _move(host, position):
    window = host.window().windowHandle()
    event = QMouseEvent(QEvent.MouseMove, QPointF(position),
                        QPointF(host.mapToGlobal(position)), Qt.NoButton,
                        Qt.NoButton, Qt.NoModifier)
    QApplication.sendEvent(window, event)


def test_a_hover_move_sets_the_gravity_pointer_without_a_tick(qtbot):
    host, field = _field(qtbot, gravity_radius=0.1)
    assert field._interaction_app is not None
    _move(host, QPoint(90, 60))
    with field._engine_lock:
        field._art_input._consume(field.engine)
    assert field.engine.pointer == pytest.approx(
        ((90 + .5) / field.width(), (60 + .5) / field.height()), abs=.01)


def test_a_move_wakes_the_shading_thread_and_the_frame_is_painted_promptly(qtbot):
    host, field = _field(qtbot, gravity_radius=0.1)
    producer = field._producer_box[0]
    assert producer is not None
    producer.set_fps(1)
    qtbot.wait(50)
    shaded, painted = producer.frames_shaded, field.frames_painted
    started = time.perf_counter()
    _move(host, QPoint(200, 120))
    qtbot.waitUntil(lambda: producer.frames_shaded > shaded
                    and field.frames_painted > painted, timeout=500)
    assert time.perf_counter() - started < 0.4


def test_gravity_moves_do_not_wake_shading_during_a_run(qtbot):
    host, field = _field(qtbot, gravity_radius=0.1)
    producer = field._producer_box[0]
    producer.set_fps(1)
    field._run_paced = True
    qtbot.wait(50)
    shaded = producer.frames_shaded
    _move(host, QPoint(150, 100))
    qtbot.wait(150)
    assert producer.frames_shaded == shaded


def test_hover_moves_drive_gravity_on_other_interactive_art(qtbot):
    host, field = _field(qtbot, theme="data_art_point_atlas", gravity_radius=0.2)
    assert field._interaction_app is not None
    _move(host, QPoint(60, 60))
    with field._engine_lock:
        field._art_input._consume(field.engine)
    assert field.engine.pointer is not None
    field.set_gravity_radius(0)
    assert field._interaction_app is None
