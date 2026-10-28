"""Item 694: the photomosaic startup screen and its blow-away.

The loading cover shows the spaCR logo as a photomosaic of microscopy tiles
while the window loads; when the window is ready the tiles lift, turn over,
spin and fly off downwind, revealing the window. These tests drive the cover
with a fake clock, so every frame is reproducible and no test waits for the
animation in real time.

Item 695 changed the timing (a 2 s hold, a 4 s blow-away, the container
fading from 1 to 3 s into it), the logo's size (half the screen height) and
its resource (2160 px from the 17920 px picture); these tests follow it, and
tests/qt/test_695_startup_sequence_and_borderless_window.py tests the rest.
"""
from __future__ import annotations

import json
import math
import os
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPointF, QRect, QSize, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QImageReader, QKeyEvent, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from spacr.qt.widgets import loading_screen as ls  # noqa: E402

pytestmark = pytest.mark.qt


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 50.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def host(qapp):
    widget = QWidget()
    widget.resize(900, 700)
    yield widget
    widget.deleteLater()
    QApplication.processEvents()


@pytest.fixture()
def cover(host):
    screen = ls.LoadingScreen(total=4, parent=host)
    clock = _Clock()
    screen._clock = clock
    screen.clock = clock
    assert screen.show_mosaic()
    return screen


def _played(screen, *, painted=True):
    """Start ``screen``'s blow-away past its hold, on its fake clock."""
    if painted:
        screen.grab()
    assert screen.blow_away(force=True)
    screen.clock.now += ls._MOSAIC_MIN_HOLD_S
    screen._begin()
    assert screen._state == "playing"
    return screen


# --- the resource ---------------------------------------------------------

def test_the_packaged_mosaic_is_the_80_by_80_grid_of_the_generator():
    """color_final_v1: G = 80 tiles, each resampled on its own to tile_px."""
    mosaic = ls.MosaicMap.load()
    assert mosaic is not None
    assert mosaic.grid == 80
    image = QImage(mosaic.image_path)
    assert image.width() == image.height() == mosaic.grid * mosaic.tile_px
    assert 3000 < int(mosaic.occupied.sum()) < 6400
    assert os.path.getsize(mosaic.image_path) < 1_000_000


def test_the_tile_map_matches_the_tiles_in_the_picture(qapp):
    """Logo tiles are bright and background tiles dark, cell by cell."""
    mosaic = ls.MosaicMap.load()
    image = QImage(mosaic.image_path).convertToFormat(QImage.Format_RGB888)
    side = image.width()
    pixels = np.frombuffer(image.constBits(), np.uint8).reshape(
        side, image.bytesPerLine())[:, :side * 3].reshape(side, side, 3)
    tile = mosaic.tile_px
    mean = pixels.reshape(mosaic.grid, tile, mosaic.grid, tile, 3).mean(
        axis=(1, 3, 4))
    background = np.median(mean[~mosaic.occupied])
    assert np.median(mean[mosaic.occupied]) > 2 * background
    assert np.percentile(mean[~mosaic.occupied], 95) < np.percentile(
        mean[mosaic.occupied], 25)


def test_a_missing_or_broken_map_leaves_the_cover_plain(tmp_path, host,
                                                        monkeypatch):
    assert ls.MosaicMap.load(str(tmp_path)) is None
    (tmp_path / "photomosaic.json").write_text(json.dumps(
        {"grid": 2, "tile_px": 1, "image": "gone.jpg", "occupied": "c0"}), encoding="utf-8")
    assert ls.MosaicMap.load(str(tmp_path)) is None
    (tmp_path / "gone.jpg").write_bytes(b"")
    assert ls.MosaicMap.load(str(tmp_path)) is not None
    (tmp_path / "photomosaic.json").write_text(json.dumps(
        {"grid": 9, "tile_px": 1, "image": "gone.jpg", "occupied": "c0"}), encoding="utf-8")
    assert ls.MosaicMap.load(str(tmp_path)) is None
    (tmp_path / "photomosaic.json").write_text("{not json", encoding="utf-8")
    assert ls.MosaicMap.load(str(tmp_path)) is None

    monkeypatch.setattr(ls.MosaicMap, "load", classmethod(lambda cls: None))
    screen = ls.LoadingScreen(parent=host)
    assert screen.show_mosaic() is False
    assert screen.mosaic_shown() is False
    assert screen.blow_away(force=True) is False


# --- the switches ---------------------------------------------------------

def test_the_environment_switch(monkeypatch):
    monkeypatch.delenv(ls.NO_SPLASH_ENV, raising=False)
    assert ls.mosaic_splash_wanted()
    for off in ("0", "false", "no", "off", ""):
        monkeypatch.setenv(ls.NO_SPLASH_ENV, off)
        assert ls.mosaic_splash_wanted()
    monkeypatch.setenv(ls.NO_SPLASH_ENV, "1")
    assert not ls.mosaic_splash_wanted()
    assert not ls.startup_animation_allowed()


def test_offscreen_never_animates(monkeypatch):
    monkeypatch.delenv(ls.NO_SPLASH_ENV, raising=False)
    assert QApplication.platformName() == "offscreen"
    assert ls.startup_animation_allowed() is False


@pytest.fixture()
def on_a_display(monkeypatch):
    from PySide6.QtGui import QGuiApplication

    monkeypatch.delenv(ls.NO_SPLASH_ENV, raising=False)
    monkeypatch.setattr(QGuiApplication, "platformName",
                        staticmethod(lambda: "xcb"))


def test_reduce_motion_and_the_preference_turn_it_off(qapp, on_a_display):
    from spacr.qt import preferences

    assert preferences.get_startup_animation() is True
    assert ls.startup_animation_allowed() is True
    preferences.set_startup_animation(False)
    assert preferences.get_startup_animation() is False
    assert ls.startup_animation_allowed() is False
    preferences.set_startup_animation(True)
    preferences.set_reduce_motion(True)
    assert ls.startup_animation_allowed() is False
    preferences.set_reduce_motion(False)
    assert ls.startup_animation_allowed() is True


def test_an_unreadable_preference_counts_as_off(qapp, on_a_display,
                                                monkeypatch):
    from spacr.qt import preferences

    def broken():
        raise RuntimeError("store gone")

    monkeypatch.setattr(preferences, "get_startup_animation", broken)
    assert ls.startup_animation_allowed() is False


def test_will_blow_away_needs_a_mosaic_cover_and_permission(
        cover, on_a_display):
    assert ls.will_blow_away(object()) is False
    assert ls.will_blow_away(cover) is True
    cover.skip()
    assert ls.will_blow_away(cover) is False
    plain = ls.LoadingScreen(parent=cover.parent())
    assert ls.will_blow_away(plain) is False


def test_without_permission_the_cover_is_left_to_its_owner(cover):
    assert cover.blow_away() is False
    assert cover._state == "idle"


# --- the still picture ----------------------------------------------------

def test_the_mosaic_is_painted_centred_above_the_strap_line(cover):
    shot = cover.grab().toImage()
    x, y, side, tile, ratio = cover._mosaic_layout()
    assert side * ratio <= tile * 80 < side * ratio + 80
    assert abs((x + side / 2.0) - cover.width() / 2.0) < 1.0
    def green_share(x0, y0, x1, y1):
        hits = total = 0
        # A stride aliases the downscaled microscopy tile grid; count every pixel.
        for py in range(int(y0), int(y1)):
            for px in range(int(x0), int(x1)):
                total += 1
                hits += shot.pixelColor(px, py).green() > 80
        return hits / max(1, total)

    assert green_share(x, y, x + side, y + side) > 0.03
    assert green_share(0, y, x - 2, y + side) == 0.0
    assert cover._first_paint_at == cover.clock.now
    picture = cover._mosaic_picture(tile, ratio)
    assert picture is cover._mosaic_picture(tile, ratio), "cached"


def test_an_unreadable_picture_falls_back_to_the_mark(cover, tmp_path):
    cover._mosaic.image_path = str(tmp_path / "missing.jpg")
    cover.grab()
    assert cover.mosaic_shown() is False


def test_black_becomes_transparent_and_colour_survives(qapp):
    image = QImage(4, 2, QImage.Format_RGB32)
    image.fill(QColor(0, 0, 0))
    image.setPixelColor(1, 1, QColor(20, 220, 60))
    out = ls._black_to_alpha(image)
    assert out.pixelColor(0, 0).alpha() == 0
    lit = out.pixelColor(1, 1)
    assert lit.alpha() == 220
    assert lit.green() == 255
    assert abs(lit.red() - 23) <= 1 and abs(lit.blue() - 69) <= 1


# --- the motion -----------------------------------------------------------

def _model(**kwargs):
    occupied = np.zeros((80, 80), bool)
    occupied[20:60, 20:60] = True
    return ls.BlowAway(occupied, (100.0, 50.0), 400.0, **kwargs)


def test_the_front_reaches_upwind_tiles_first():
    model = _model()
    launch = model.launch.reshape(80, 80)
    assert launch[:, :10].mean() < launch[:, 35:45].mean() < \
        launch[:, 70:].mean()
    assert model.launch.min() >= model.start
    assert model.launch.max() <= model.start + model.sweep
    assert 1.5 <= model.duration <= 2.2


def test_a_reversed_wind_starts_from_the_other_side():
    model = _model(wind=(-1.0, -0.3))
    launch = model.launch.reshape(80, 80)
    assert launch[:, :10].mean() > launch[:, 70:].mean()


def test_the_motion_depends_on_time_not_on_frames():
    once = _model().frame(0.9)
    stepped = _model()
    for t in np.linspace(0.0, 0.9, 37):
        stepped.frame(float(t))
    again = stepped.frame(0.9)
    for left, right in zip(once, again):
        np.testing.assert_array_equal(left, right)
    other_seed = _model(seed=1).frame(0.9)
    assert not np.array_equal(once[5], other_seed[5])


def test_a_tile_lifts_flips_spins_flies_downwind_and_fades():
    model = _model()
    first = int(np.argmin(model.fly_launch))
    start = float(model.fly_launch[first])
    index, m11, m12, m21, m22, dx, dy, alpha = model.frame(start + 1e-4)
    at = int(np.flatnonzero(index == first)[0])
    assert abs(m11[at] - 1.0) < 0.01 and abs(m22[at] - 1.0) < 0.01
    assert abs(dx[at] - model.centre_x[first]) < 0.5
    assert alpha[at] > 0.95

    later = start + 0.3
    index, m11, m12, m21, m22, dx, dy, alpha = model.frame(later, floor=0.0)
    at = int(np.flatnonzero(index == first)[0])
    determinant = m11[at] * m22[at] - m12[at] * m21[at]
    flip = model.flip_rate[first] * (0.3 - 0.03)
    assert np.sign(determinant) == np.sign(np.cos(flip))
    assert dx[at] > model.centre_x[first] + 0.05 * model.side
    assert alpha[at] < 1.0

    gone = model.frame(start + float(model.life[first]) + 0.01)[0]
    assert first not in set(gone.tolist())


def test_faint_and_offscreen_tiles_are_not_drawn():
    model = _model()
    everything = model.frame(1.2, floor=0.0)[0].size
    some = model.frame(1.2, floor=0.5)[0].size
    inside = model.frame(1.2, (0.0, 0.0, 300.0, 300.0), 0.0)[0].size
    assert everything > some > 0
    assert everything > inside


def test_the_ground_clears_behind_the_front_and_then_everywhere():
    model = _model()
    front, behind, ahead = model.backdrop(0.0)
    assert behind == pytest.approx(1.0) and ahead == pytest.approx(1.0)
    front, behind, ahead = model.backdrop(model.start + 0.6)
    assert behind < 0.05 < ahead
    assert 0.4 < front < 0.7
    front, behind, ahead = model.backdrop(model.duration)
    assert behind == ahead == 0.0
    assert model.launched(model.duration).size == 80 * 80
    assert model.launched(0.0).size == 0


def test_an_empty_mosaic_still_has_a_length():
    model = ls.BlowAway(np.zeros((4, 4), bool), (0, 0), 40.0)
    assert model.flyers.size == 0
    assert model.duration == pytest.approx(model.start + model.sweep + 0.52)


# --- the blow-away on the cover -------------------------------------------

def test_the_blow_away_waits_for_the_hold_then_plays_and_finishes(
        cover, qtbot):
    finished = []
    cover.grab()
    assert cover.blow_away(lambda: finished.append(True), force=True)
    assert cover._state == "holding"
    assert cover.playing()
    assert cover.blow_away(force=True) is False, "only once"
    cover._begin()
    assert cover._state == "holding", "the hold is not skipped"
    cover.clock.now += ls._MOSAIC_MIN_HOLD_S
    cover._begin()
    assert cover._state == "playing"
    assert cover.elapsed() == 0.0
    assert not cover.testAttribute(Qt.WA_OpaquePaintEvent)
    assert not cover.autoFillBackground()
    assert len(cover._tiles) == cover._model.flyers.size
    timer = cover._tick_timer
    assert timer is not None and timer.parent() is cover
    assert timer.isSingleShot() and timer.isActive()
    budget = ls._frame_interval_ms(ls._screen_refresh_hz(cover.screen()))
    assert cover._frame_budget_ms() == pytest.approx(budget)
    assert 0 <= timer.interval() <= math.ceil(budget)

    for step in range(4):
        cover.clock.now += 0.4
        cover._tick()
        cover.grab()
    assert cover._rest_cleared > 0
    stats = cover.frame_stats()
    assert stats["frames"] == 4 and stats["most_tiles"] > 0
    assert stats["max_ms"] >= stats["mean_ms"] > 0.0

    cover.clock.now += cover._model.duration
    cover._tick()
    assert finished == [True]
    assert cover._state == "done"
    assert cover._tick_timer is None
    assert cover.isHidden()
    assert cover not in ls._PLAYING


def test_the_hold_counts_from_the_first_paint(cover, qtbot):
    calls = []
    cover._begin = lambda: calls.append(cover.clock.now)
    assert cover.blow_away(force=True)
    assert cover._state == "waiting", "nothing painted yet"
    cover.grab()
    assert cover._state == "holding"
    qtbot.waitUntil(lambda: bool(calls), timeout=3000)


def test_a_late_call_after_the_hold_starts_at_once(cover, qtbot):
    calls = []
    cover.grab()
    cover.clock.now += 5.0
    cover._begin = lambda: calls.append(True)
    assert cover.blow_away(force=True)
    qtbot.waitUntil(lambda: bool(calls), timeout=1000)


def test_begin_only_runs_once_it_is_holding(cover):
    cover._begin()
    assert cover._state == "idle"


def test_a_picture_that_vanishes_before_the_start_ends_the_cover(
        cover, tmp_path):
    cover.grab()
    assert cover.blow_away(force=True)
    cover._mosaic_key = None
    cover._mosaic.image_path = str(tmp_path / "missing.jpg")
    cover.clock.now += ls._MOSAIC_MIN_HOLD_S
    cover._begin()
    assert cover._state == "done"


def test_a_click_or_a_key_skips_to_the_end(cover, qapp):
    _played(cover)
    assert ls.blow_away_running()
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5),
                        QPointF(5, 5), Qt.LeftButton, Qt.LeftButton,
                        Qt.NoModifier)
    cover.mousePressEvent(press)
    assert cover._state == "done"
    assert not ls.blow_away_running()

    other = ls.LoadingScreen(parent=cover.parent())
    other._clock = cover.clock
    other.clock = cover.clock
    other.show_mosaic()
    _played(other)
    assert other.focusPolicy() == Qt.FocusPolicy.StrongFocus
    key = QKeyEvent(QEvent.Type.KeyPress, Qt.Key_Escape, Qt.NoModifier)
    other.keyPressEvent(key)
    assert key.isAccepted()
    assert other._state == "done"
    assert other.focusPolicy() == Qt.FocusPolicy.NoFocus


def test_the_focus_goes_back_where_it_was(host, qtbot):
    from PySide6.QtWidgets import QLineEdit

    field = QLineEdit(host)
    host.show()
    qtbot.waitExposed(host)
    host.activateWindow()
    field.setFocus()
    screen = ls.LoadingScreen(parent=host)
    screen.setGeometry(host.rect())
    screen.show()
    screen.show_mosaic()
    screen._clock = screen.clock = _Clock()
    screen.grab()
    before = QApplication.focusWidget()
    assert screen.blow_away(force=True)
    assert screen._focus_before is before
    screen._finish()
    assert screen._focus_before is None
    assert QApplication.focusWidget() is not screen


def test_a_focus_owner_deleted_meanwhile_is_let_go(cover):
    class _Gone:
        def isVisible(self):
            raise RuntimeError("deleted")

    cover._focus_before = _Gone()
    cover._give_the_keys_back()
    assert cover._focus_before is None


def test_a_cover_deleted_while_playing_stops_counting(host):
    from shiboken6 import delete

    screen = ls.LoadingScreen(parent=host)
    screen.show_mosaic()
    screen._clock = screen.clock = _Clock()
    screen.grab()
    assert screen.blow_away(force=True)
    assert not ls.blow_away_running(), "the window loads while it holds"
    screen.clock.now += ls._MOSAIC_MIN_HOLD_S
    screen._begin()
    assert ls.blow_away_running()
    delete(screen)
    assert not ls.blow_away_running()
    assert not list(ls._PLAYING)


class _Window(QWidget):
    """Just enough window for a screen prewarm."""

    _opening_a_screen = False

    def __init__(self):
        super().__init__()
        from PySide6.QtWidgets import QStackedWidget

        self._stack = QStackedWidget(self)
        self._startup = QWidget(self._stack)
        self._stack.addWidget(self._startup)

    def _prewarmable_keys(self):
        return []


def test_idle_warming_waits_for_the_animation(cover, qtbot):
    from spacr.qt import app as qt_app
    from spacr.qt import preferences

    window = _Window()
    window.show()
    qtbot.addWidget(window)
    qtbot.waitExposed(window)
    prewarm = qt_app._ScreenPrewarm(window, [])
    prewarm._last_input = qt_app._time_monotonic() - 600.0
    assert preferences._the_user_is_busy(window) is False
    assert prewarm.busy_reason() == ""
    _played(cover)
    assert preferences._the_user_is_busy(window) is True
    assert prewarm.busy_reason() == "startup animation"
    cover._finish()
    assert preferences._the_user_is_busy(window) is False
    assert prewarm.busy_reason() == ""


def test_other_states_ignore_a_skip(cover):
    _played(cover)
    cover._finish()
    cover.skip()
    assert cover._state == "done"


def test_a_click_on_the_cover_during_loading_skips_the_animation(cover):
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5),
                        QPointF(5, 5), Qt.LeftButton, Qt.LeftButton,
                        Qt.NoModifier)
    cover.mousePressEvent(press)
    assert press.isAccepted()
    assert cover._skip_requested
    assert cover.blow_away(force=True) is False


def test_skip_while_holding_ends_without_frames(cover):
    cover.grab()
    assert cover.blow_away(force=True)
    cover.skip()
    assert cover._state == "done"
    stats = cover.frame_stats()
    assert {k: stats[k] for k in ("frames", "mean_ms", "max_ms",
                                  "most_tiles")} == {
        "frames": 0, "mean_ms": 0.0, "max_ms": 0.0, "most_tiles": 0}
    assert stats["interval_p95_ms"] == stats["longest_gap_ms"] == 0.0
    assert stats["dropped_frames"] == stats["late_frames"] == 0
    cover.skip()
    cover._finish()
    assert cover._state == "done"


def test_no_leftover_timers_filters_or_widgets(host, qtbot):
    screen = ls.LoadingScreen(parent=host)
    screen.show_mosaic()
    screen._clock = screen.clock = _Clock()
    _played(screen)
    screen.clock.now += 10.0
    screen._tick()
    qtbot.waitUntil(lambda: not host.findChildren(ls.LoadingScreen),
                    timeout=2000)
    from PySide6.QtCore import QTimer

    assert not [t for t in host.findChildren(QTimer) if t.isActive()]


def test_slow_frames_shed_faint_tiles_and_fast_ones_take_them_back(cover):
    cover._adapt(20.0)
    assert cover._floor == pytest.approx(0.10)
    cover._adapt(12.0)
    assert cover._floor == pytest.approx(0.10)
    cover._adapt(5.0)
    assert cover._floor == pytest.approx(0.06)
    for _ in range(20):
        cover._adapt(50.0)
    assert cover._floor == 0.5
    for _ in range(40):
        cover._adapt(1.0)
    assert cover._floor == 0.02


def test_clearing_twice_at_one_moment_does_nothing_new(cover):
    _played(cover)
    cover._clear_launched(0.8)
    cleared = cover._rest_cleared
    cover._clear_launched(0.8)
    assert cover._rest_cleared == cleared > 0
    cover._finish()


def test_sixty_frames_a_second_is_within_reach(cover):
    """The heaviest frame of the flight, drawn for real, offscreen."""
    cover.parent().resize(1600, 1000)
    cover.setGeometry(cover.parent().rect())
    _played(cover)
    for t in np.linspace(0.2, 1.6, 15):
        cover.clock.now = cover._began_at + float(t)
        cover.grab()
    stats = cover.frame_stats()
    # Retina Macs combine each 2x2 block into one flying pixmap.
    assert stats["most_tiles"] > (250 if sys.platform == "darwin" else 1000)
    assert stats["mean_ms"] < 40.0
    cover._finish()


# --- the window -----------------------------------------------------------

def test_the_window_shows_the_mosaic_unless_told_not_to(qapp, monkeypatch):
    import PySide6.QtWidgets as qtwidgets

    from spacr.qt import app as qt_app

    real = qtwidgets.QApplication

    class _OnScreen:
        @staticmethod
        def instance():
            live = real.instance()

            class _Says:
                def __getattr__(self, name):
                    return getattr(live, name)

                @staticmethod
                def platformName():
                    return "xcb"
            return _Says()

    monkeypatch.setattr(qtwidgets, "QApplication", _OnScreen)
    monkeypatch.delenv(ls.NO_SPLASH_ENV, raising=False)
    host = QWidget()
    try:
        screen = qt_app.MainWindow._install_loading_screen(host)
        assert screen.mosaic_shown()
        monkeypatch.setenv(ls.NO_SPLASH_ENV, "1")
        plain = qt_app.MainWindow._install_loading_screen(host)
        assert not plain.mosaic_shown()
    finally:
        host.deleteLater()


def test_the_preload_ending_hands_the_cover_to_the_blow_away(
        cover, on_a_display):
    from spacr.qt import app as qt_app

    host = cover.parent()
    host._loading_screen = cover
    released = []
    host._release_the_startup_dialogs = lambda: released.append(1)
    cover.grab()
    qt_app.MainWindow._on_preload_done(host)
    assert host._loading_screen is None
    assert host._blowing_cover is cover and released == []
    assert cover._state == "holding"
    cover._finish()


def test_a_lazy_window_keeps_a_mosaic_cover_and_blows_it_away(
        qtbot, qt_theme_applied, monkeypatch, on_a_display):
    from spacr.qt import app as app_mod
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_preload_policy", lambda: "lazy")
    made = []

    def install(self):
        screen = ls.LoadingScreen(parent=self)
        screen.show_mosaic()
        made.append(screen)
        return screen

    monkeypatch.setattr(app_mod.MainWindow, "_install_loading_screen",
                        install)
    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    assert window._loading_screen is made[0]
    qtbot.waitUntil(lambda: window._loading_screen is None, timeout=3000)
    assert made[0]._state in ("waiting", "holding")
    made[0]._finish()


# --- Preferences ----------------------------------------------------------

def test_the_preferences_switch_saves_and_resets(qtbot, qt_theme_applied):
    from PySide6.QtWidgets import QAbstractButton, QDialogButtonBox

    from spacr.qt import preferences

    dialog = preferences.PreferencesDialog(None)
    qtbot.addWidget(dialog)
    switch = dialog.findChild(QAbstractButton, "StartupAnimation")
    assert switch is not None and switch.isChecked()
    assert switch.accessibleName() == "Startup animation"
    switch.setChecked(False)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert preferences.get_startup_animation() is False

    again = preferences.PreferencesDialog(None)
    qtbot.addWidget(again)
    switch = again.findChild(QAbstractButton, "StartupAnimation")
    assert not switch.isChecked()
    from PySide6.QtWidgets import QPushButton

    again.findChild(QPushButton, "PreferencesReset").click()
    assert switch.isChecked(), "Reset puts the default back: on"
    again.reject()


# --- the remaining branches -----------------------------------------------

def test_nothing_has_elapsed_before_the_start(cover):
    assert cover.elapsed() == 0.0


def test_a_map_can_be_handed_in(host):
    mosaic = ls.MosaicMap.load()
    screen = ls.LoadingScreen(parent=host)
    assert screen.show_mosaic(mosaic)
    assert screen._mosaic is mosaic


class _Holder:
    """Stands in for the widget that had the keyboard focus."""

    def __init__(self, visible):
        self.visible = visible
        self.focused = 0

    def isVisible(self):
        return self.visible

    def setFocus(self, _reason):
        self.focused += 1


def test_the_focus_is_returned_only_to_a_visible_widget(cover):
    shown, hidden = _Holder(True), _Holder(False)
    cover._focus_before = shown
    cover._give_the_keys_back()
    cover._focus_before = hidden
    cover._give_the_keys_back()
    assert (shown.focused, hidden.focused) == (1, 0)


def test_an_idle_cover_in_the_set_does_not_count(cover):
    ls._PLAYING.add(cover)
    try:
        assert not ls.blow_away_running()
    finally:
        ls._PLAYING.discard(cover)


@pytest.mark.parametrize("tile,ratio", [(3, 1.0), (5, 1.5), (7, 2.0)])
@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_only_logo_tiles_survive_decoding_and_cutting(cover, monkeypatch,
                                                      tile, ratio, platform):
    mosaic = cover._mosaic
    reader = QImageReader(mosaic.image_path)
    reader.setScaledSize(QSize(tile * mosaic.grid, tile * mosaic.grid))
    original = ls._black_to_alpha(reader.read())
    source = np.frombuffer(original.bits(), dtype=np.uint32).reshape(
        original.height(), original.bytesPerLine() // 4)[:, :original.width()]
    eligible = np.repeat(np.repeat(mosaic.occupied, tile, axis=0), tile, axis=1)
    assert np.any(source[~eligible]), "The original contains background texture"
    picture = cover._mosaic_picture(tile, ratio)
    image = picture.toImage().convertToFormat(QImage.Format_ARGB32_Premultiplied)
    pixels = np.frombuffer(image.bits(), dtype=np.uint32).reshape(
        image.height(), image.bytesPerLine() // 4)[:, :image.width()]
    assert not np.any(pixels[~eligible])
    np.testing.assert_array_equal(pixels[eligible], source[eligible])
    assert picture is cover._mosaic_picture(tile, ratio)
    monkeypatch.setattr(ls, "sys", types.SimpleNamespace(platform=platform))
    cover._cut_tiles(picture, 0.0, 0.0, 400.0, tile)
    rest = cover._rest
    resting = np.frombuffer(rest.bits(), dtype=np.uint32).reshape(
        rest.height(), rest.bytesPerLine() // 4)[:, :rest.width()]
    np.testing.assert_array_equal(resting, pixels)
    cells = mosaic.occupied
    if platform == "darwin":
        cells = cells.reshape(40, 2, 40, 2).any(axis=(1, 3))
    np.testing.assert_array_equal(cover._model.flyers, np.flatnonzero(cells))
    for flat, piece in zip(cover._model.flyers, cover._tiles):
        row, col = divmod(int(flat), cover._model.grid)
        pitch = cover._rest_tile
        assert piece.toImage() == rest.copy(QRect(col * pitch, row * pitch,
                                                  pitch, pitch))


@pytest.mark.parametrize("plain", [None, False, True],
                         ids=["static", "blow-away", "reduced-motion"])
def test_logo_outline_and_empty_background_survive_replay(qtbot, host,
                                                          tmp_path, plain):
    path = tmp_path / "mosaic.png"
    image = QImage(16, 16, QImage.Format_RGB32)
    image.fill(QColor(80, 120, 160))
    assert image.save(str(path))
    occupied = np.array([[0, 1, 1, 0], [1, 0, 0, 1],
                         [1, 0, 0, 1], [0, 1, 1, 0]], dtype=bool)
    for replay in range(2):
        screen = ls.LoadingScreen(parent=host)
        qtbot.addWidget(screen)
        clock = _Clock()
        screen._clock = clock
        assert screen.show_mosaic(ls.MosaicMap(str(path), 4, 4, occupied))
        screen.grab()
        if plain is not None:
            assert screen.blow_away(force=True, plain=plain)
            clock.now += ls._MOSAIC_MIN_HOLD_S
            screen._begin()
            assert screen._state == "playing"
            assert screen._plain is plain
        shot = screen.grab().toImage()
        x, y, side, _tile, _ratio = screen._mosaic_layout()
        ground = shot.pixelColor(0, 0)
        for row, col in np.ndindex(occupied.shape):
            pixel = shot.pixelColor(int(x + (col + 0.5) * side / 4),
                                    int(y + (row + 0.5) * side / 4))
            assert (pixel != ground) is bool(occupied[row, col]), (replay, row, col)
        if plain is not None:
            screen._finish()
        else:
            screen.close()
