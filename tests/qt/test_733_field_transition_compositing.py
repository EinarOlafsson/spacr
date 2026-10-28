"""Navigator geometry and idle-release layers leave no stale foreground pixels."""
import math
import time

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import QColor, QImage, QRegion
from PySide6.QtWidgets import QApplication, QWidget

from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.fixture(autouse=True)
def _software_rendering_only(monkeypatch):
    monkeypatch.setenv('SPACR_NO_GL', '1')


def test_fade_layer_clears_pixels_outside_each_current_partial_region(qtbot):
    from spacr.qt.widgets import ambient

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 240)
    backdrop = ambient.AmbientWidget(host, theme='data_art_impulse_lens', seed=7)
    backdrop.setGeometry(host.rect())
    held = backdrop._held_rows
    ratio = backdrop.devicePixelRatioF()
    held._layer = QImage(math.ceil(320 * ratio), math.ceil(240 * ratio),
                         QImage.Format_ARGB32_Premultiplied)
    held._layer.setDevicePixelRatio(ratio)
    held._layer.fill(QColor(230, 20, 180))
    frame = QImage(320, 240, QImage.Format_ARGB32_Premultiplied)
    frame.fill(QColor(0, 30, 90))

    def color(image, x, y):
        return image.pixelColor(int(x * ratio), int(y * ratio))

    first = held.layer(backdrop, frame, QRegion(QRect(60, 40, 100, 80)))
    assert color(first, 80, 60).alpha() == 255
    assert color(first, 10, 10).alpha() == 0
    second = held.layer(backdrop, frame, QRegion(QRect(200, 140, 80, 50)))
    assert first is second
    assert color(second, 220, 160).alpha() == 255
    assert color(second, 80, 60).alpha() == 0
    assert color(second, 10, 10).alpha() == 0


def test_card_geometry_clear_drops_old_hold_without_forcing_particle_ring(
        qtbot, private_ai_profile):
    from spacr.qt.app import MainWindow
    from spacr.qt.widgets import ambient

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1000, 760)
    window.show()
    try:
        qtbot.waitUntil(lambda: window.window_backdrop() is not None)
        backdrop = window.window_backdrop()
        qtbot.waitUntil(lambda: backdrop._last_frame is not None)
        page = window._startup
        old = ((.35, .35, .65, .65),)
        backdrop._art_input._growing_regions = old
        with backdrop._engine_lock:
            backdrop._art_input._consume(backdrop._engine)
        held = backdrop._held_rows
        held.region = QRegion(QRect(100, 100, 200, 200))
        held.frame = backdrop._last_frame
        page.field.set_cards([('mask', 'Mask', 'Detected image.', '')])
        assert backdrop._art_input._growing_regions == ()
        assert held.frame is None and held.fade is None
        assert any(card._growing for card in page.field.cards)
        with backdrop._engine_lock:
            backdrop._art_input._consume(backdrop._engine)
            assert backdrop._engine._growing_regions == ()
            x = np.linspace(.3, .7, 100, dtype=np.float32)
            y = np.full_like(x, .5)
            actual_x, actual_y = backdrop._engine._part_growing_regions(x, y, 1000, 760)
            np.testing.assert_array_equal(actual_x, x)
            np.testing.assert_array_equal(actual_y, y)
        held._stirred = 123.0
        page.field._publish_growth()
        assert held._stirred == 123.0
    finally:
        window.close()


def test_real_idle_mouse_release_does_not_replay_displaced_window_text(
        qtbot, private_ai_profile, monkeypatch):
    from spacr.qt.app import MainWindow, install_the_dialog_filters

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_ambient_enabled(True)
    preferences.set_reduce_motion(False)
    preferences.set_theme('dark')
    preferences.set_ambient_theme('data_art_impulse_lens')
    preferences.set_ambient_density(.25)
    preferences.apply_preferences_to_app(QApplication.instance())
    window = MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    window.resize(1100, 820)
    window.show()
    install_the_dialog_filters(QApplication.instance())
    try:
        qtbot.waitUntil(lambda: window.window_backdrop() is not None)
        backdrop = window.window_backdrop()
        qtbot.waitUntil(lambda: backdrop._last_frame is not None)
        page = window._startup
        page.field.set_cards([(key, key, 'Detected local data.', '')
                              for key in ('mask', 'make_masks', 'convert')])
        qtbot.waitUntil(lambda: all(not card._growing for card in page.field.cards), timeout=7000)
        producer = backdrop._producer_box[0]
        assert producer is not None
        backdrop._held_rows.release(fade=False)
        backdrop._held_rows._stirred = time.monotonic()
        native_window = window.windowHandle()
        resting_point = page.mapToGlobal(QPoint(page.width() // 2, 40))
        assert not window._dock_edge.rect().contains(window._dock_edge.mapFromGlobal(resting_point))
        qtbot.mouseMove(native_window, native_window.mapFromGlobal(resting_point))
        qtbot.wait(3100)
        held = backdrop._held_rows
        assert held.frame is None and not held.arming
        assert held.region.isEmpty() and held.live.isEmpty()
        assert held._covered().isEmpty()
        frame = backdrop._last_frame.copy()
        monkeypatch.setattr(producer, 'latest', lambda: frame)
        backdrop._last_frame = frame
        backdrop.repaint()

        def pixels():
            image = window.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
            return np.frombuffer(image.constBits(), np.uint8).reshape(
                image.height(), image.bytesPerLine() // 4, 4)[:, :image.width(), :3].copy()

        before = pixels()
        allowed = QRegion()
        for widget in [page.path_edit, page.field.details, *page.field.cards]:
            allowed |= QRegion(QRect(widget.mapTo(window, QPoint()), widget.size()).adjusted(-3, -3, 3, 3))
        card = page.field.cards[0]
        held = backdrop._held_rows
        releases = []
        original_release = held.release

        def observe_release(fade=True, repaint=True):
            previous, region = held.frame, QRegion(held.region)
            original_release(fade=fade, repaint=repaint)
            if previous is not None:
                created = held.fade
                releases.append((fade, created is not None,
                                 created is not None and created[1] is previous,
                                 created is not None and created[2] == region))

        monkeypatch.setattr(held, 'release', observe_release)
        qtbot.mouseMove(native_window, native_window.mapFromGlobal(
            card.mapToGlobal(card.rect().center())))
        qtbot.wait(25)
        assert releases == []
        assert held.frame is None and held.fade is None and not held.arming
        after = pixels()
        mask = np.ones(before.shape[:2], dtype=bool)
        for rect in allowed:
            mask[max(0, rect.top()):min(mask.shape[0], rect.bottom()+1),
                 max(0, rect.left()):min(mask.shape[1], rect.right()+1)] = False
        difference = np.abs(after.astype(np.int16)-before.astype(np.int16)).max(axis=2)
        assert not np.any(difference[mask] > 1)
    finally:
        window.close()
