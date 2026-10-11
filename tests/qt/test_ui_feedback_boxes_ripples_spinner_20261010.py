"""Maintainer feedback of 2026-10-10 on boxes, ripples and the Make Masks spinner.

1. The module screens' Actions section is a rounded Card like Console and System.
2. Opening or closing a settings category sends one wave that moves the field
   only up and down its column. Preferences ripples when it opens and when
   Apply, Cancel or Save is clicked, and for nothing else.
3. A ripple is a gentle refraction: no dark hole at its source, no hard ring.
4. Make Masks has no activity spinner beside "Clear all objects".
"""
from __future__ import annotations

import numpy as np
import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QPushButton, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient

pytest.importorskip("scipy")


@pytest.fixture
def ripple_store(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "ripples.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)
    return store


def _field(qtbot, size=(480, 320)):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(*size)
    field = ambient.AmbientWidget(
        host, theme="data_art_impulse_lens", gravity_radius=0,
        popup_wave_frequency=0, blink_percent=0, density=1, seed=19)
    field.resize(host.size())
    host.show()
    field.show()
    field._timer.stop()
    return host, field


def _engine():
    return ambient.make_engine("data_art_impulse_lens", "spacr", "#191919",
                               seed=684, density=1, blink_percent=0,
                               popup_wave_frequency=0)


def _captured_points(engine, monkeypatch, size=(320, 200)):
    seen = {}
    original = engine._point_material

    def capture(width, height, x, y, light, spread=False):
        seen["x"], seen["y"] = np.array(x, copy=True), np.array(y, copy=True)
        return original(width, height, x, y, light, spread)

    monkeypatch.setattr(engine, "_point_material", capture)
    engine.shade(*size)
    return seen["x"], seen["y"]


def _grey(image):
    from PySide6.QtGui import QImage

    image = image.convertToFormat(QImage.Format_RGB888)
    raw = np.frombuffer(image.constBits(), np.uint8).reshape(
        image.height(), image.bytesPerLine())[:, :image.width() * 3]
    return raw.reshape(image.height(), image.width(), 3).astype(np.float64).mean(axis=2)


class TestTheActionsSectionIsABox:

    def test_actions_is_a_card_like_console_and_system(self, qtbot, qt_theme_applied):
        from spacr.qt.screens.app_screen import AppScreen
        from spacr.qt.widgets import Card

        screen = AppScreen("mask")
        qtbot.addWidget(screen)
        screen.resize(1400, 900)
        screen.show()
        for _ in range(10):
            QApplication.processEvents()
        section = screen._actions_section
        assert isinstance(section, Card)
        for other in (screen._console_card, screen._usage_card):
            assert type(other).__name__ in ("Card", "_CardBuiltWhenShown")
            assert section.objectName() == other.objectName() == "Card"
        assert section.title_label is screen._actions_heading
        assert section.isAncestorOf(screen._btn_run)
        assert section.isAncestorOf(screen._actions_body)
        from spacr.qt.theme import TRANSPARENT_PROPERTY

        screen._clear_page_surfaces()
        assert section.property(TRANSPARENT_PROPERTY) is not True
        screen._actions_folder.set_shut(True)
        QApplication.processEvents()
        assert not section.body.isVisibleTo(section)
        assert screen._actions_heading.isVisibleTo(section)
        screen._actions_folder.set_shut(False)


class TestCategoryRipplesAreVertical:

    def test_a_column_wave_moves_dots_only_up_and_down(self, monkeypatch):
        quiet, waved = _engine(), _engine()
        for engine in (quiet, waved):
            engine.set_time(10.0)
        waved._add_column_wave((0.1, 0.5, 0.5))
        for engine in (quiet, waved):
            engine.advance(0.5)
        qx, qy = _captured_points(quiet, monkeypatch)
        wx, wy = _captured_points(waved, monkeypatch)
        assert np.array_equal(qx, wx)
        moved = np.abs(wy - qy) > 1e-3
        assert moved.any()
        columns = wx[moved] / 320
        assert columns.min() > 0.1 - 0.06 and columns.max() < 0.5 + 0.06

    def test_a_category_toggle_sends_one_column_wave_and_no_rings(self, qtbot, ripple_store):
        from spacr.qt.widgets.section import Section

        host, field = _field(qtbot)
        section = Section("Settings", parent=host)
        section.setGeometry(40, 30, 160, 120)
        section.show()
        qtbot.mouseClick(section.header(), Qt.LeftButton)
        qtbot.waitUntil(lambda: bool(field.engine._column_waves))
        field._on_tick()
        field._timer.stop()
        assert not field.engine._popup_waves
        assert len(field.engine._column_waves) == 1


class TestPreferencesRipplesOnlyOnOpenAndItsThreeButtons:

    @pytest.fixture
    def dialog(self, qtbot, ripple_store, monkeypatch):
        host, field = _field(qtbot, size=(900, 700))
        dlg = prefs.PreferencesDialog(host)
        qtbot.addWidget(dlg)
        dlg.resize(400, 300)
        dlg.show()
        monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: dlg))
        return host, field, dlg

    def _waves(self, field):
        field._on_tick()
        field._timer.stop()
        return len(field.engine._popup_waves)

    def test_opening_ripples_once_and_nothing_recurs(self, dialog):
        _host, field, dlg = dialog
        assert dlg.property(ambient.EXPLICIT_RIPPLES_PROPERTY)
        field.engine.set_popup_wave_frequency(60.0)
        assert self._waves(field) == 1
        for _ in range(4):
            field.engine.advance(0.6)
            assert self._waves(field) == 1

    def test_a_dropdown_or_sub_dialog_inside_preferences_does_not_ripple(
            self, dialog, monkeypatch):
        _host, field, dlg = dialog
        assert self._waves(field) == 1
        dropdown = QWidget(dlg, Qt.Popup)
        dropdown.resize(120, 160)
        dropdown.show()
        monkeypatch.setattr(QApplication, "activePopupWidget",
                            staticmethod(lambda: dropdown))
        assert self._waves(field) == 1
        dropdown.hide()
        monkeypatch.setattr(QApplication, "activePopupWidget", staticmethod(lambda: None))
        assert self._waves(field) == 1

    def test_reset_and_background_clicks_do_not_ripple(self, dialog, qtbot):
        _host, field, dlg = dialog
        assert self._waves(field) == 1
        reset = dlg.findChild(QPushButton, "PreferencesReset")
        qtbot.mouseClick(reset, Qt.LeftButton)
        qtbot.mouseClick(dlg, Qt.LeftButton, pos=dlg.rect().center())
        qtbot.wait(ambient.CLICK_RIPPLE_SETTLE_MS + 60)
        assert self._waves(field) == 1

    @pytest.mark.parametrize("which", [QDialogButtonBox.Apply, QDialogButtonBox.Save,
                                       QDialogButtonBox.Cancel])
    def test_apply_cancel_and_save_each_ripple(self, dialog, monkeypatch, which):
        _host, field, dlg = dialog
        assert self._waves(field) == 1
        monkeypatch.setattr(QApplication, "activeModalWidget", staticmethod(lambda: None))
        button = dlg.findChild(QDialogButtonBox).button(which)
        button.click()
        field._on_tick()
        field._timer.stop()
        assert len(field.engine._popup_waves) == 2


class TestARippleHasNoVoid:

    @pytest.mark.parametrize("age", [0.1, 0.3, 0.6])
    def test_the_source_keeps_its_brightness_and_nothing_goes_dark(self, age):
        from scipy.ndimage import gaussian_filter

        width, height = 480, 300
        plain, waved = _engine(), _engine()
        for engine in (plain, waved):
            engine.set_time(10.0)
        waved._add_ripple((0.5, 0.5))
        for engine in (plain, waved):
            engine.advance(age)
        reference = gaussian_filter(_grey(plain.shade(width, height)), 10)
        frame = gaussian_filter(_grey(waved.shade(width, height)), 10)
        ratio = frame / np.maximum(reference, 1e-3)
        inner = ratio[20:-20, 20:-20]
        centre = ratio[height // 2 - 25:height // 2 + 25, width // 2 - 25:width // 2 + 25]
        assert 0.92 <= centre.mean() <= 1.08
        assert inner.min() >= 0.85
        assert inner.max() <= 1.2

    def test_dots_never_cross(self):
        assert ambient.RIPPLE_AMPLITUDE * ambient.RIPPLE_WAVENUMBER < 0.3

    def test_a_wave_eases_in(self, monkeypatch):
        quiet, waved = _engine(), _engine()
        for engine in (quiet, waved):
            engine.set_time(10.0)
        waved._add_ripple((0.5, 0.5))
        qx, qy = _captured_points(quiet, monkeypatch)
        wx, wy = _captured_points(waved, monkeypatch)
        assert np.allclose(qx, wx) and np.allclose(qy, wy)


class TestMakeMasksHasNoSpinner:

    def test_no_activity_spinner_beside_clear_all_objects(self, qtbot, qt_theme_applied):
        from spacr.qt.button_roles import install_button_roles
        from spacr.qt.screens import make_masks as mm
        from spacr.qt.widgets.activity_spinner import ActivitySpinner

        install_button_roles(QApplication.instance())
        screen = mm.MakeMasksScreen()
        qtbot.addWidget(screen)
        screen.resize(1400, 900)
        screen.show()
        for _ in range(20):
            QApplication.processEvents()
        assert not screen.findChildren(ActivitySpinner)
        assert getattr(screen, "_activity_spinner", None) is None
