"""Fresh profiles use 90 % GUI and font scales on every screen.

The maintainer's 2026-10-21 scaling policy supersedes the earlier small-screen
75 % and large-screen 100 % defaults. A stored choice is never replaced;
unreadable values fall back to 90 %, and Reset to defaults restores the
effective values without storing them. Mocked laptop, wide and invented
screens verify that screen geometry does not change this policy.
"""
from __future__ import annotations

import os
import tempfile

import pytest
from PySide6.QtCore import QRect, QSettings
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QPushButton, QSlider


class _Screen:
    def __init__(self, width, height):
        self._geometry = QRect(0, 0, width, height)

    def availableGeometry(self):
        return self._geometry


@pytest.fixture()
def private_store(monkeypatch):
    from spacr.qt import preferences

    path = os.path.join(tempfile.mkdtemp(prefix="spacr-prefs-"), "user.ini")
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))
    return path


@pytest.fixture()
def screen(monkeypatch, qapp):
    def use(width, height):
        monkeypatch.setattr(QGuiApplication, "platformName", staticmethod(lambda: "xcb"))
        monkeypatch.setattr(QGuiApplication, "primaryScreen",
                            staticmethod(lambda: _Screen(width, height)))
    return use


def test_a_13_inch_laptop_starts_at_90_percent(private_store, screen):
    from spacr.qt import preferences

    screen(1440, 875)
    assert preferences.get_gui_scale() == pytest.approx(0.9)
    assert preferences.get_font_scale() == pytest.approx(0.9)


def test_a_1920_wide_screen_starts_at_90_percent(private_store, screen):
    from spacr.qt import preferences

    screen(1920, 1040)
    assert preferences.get_gui_scale() == pytest.approx(0.9)
    assert preferences.get_font_scale() == pytest.approx(0.9)


@pytest.mark.parametrize("stored", [1.0, 1.25, 0.5])
def test_a_stored_scale_is_never_overridden(private_store, screen, stored):
    from spacr.qt import preferences

    preferences.set_gui_scale(stored)
    preferences.set_font_scale(stored)
    for width in (1440, 1920):
        screen(width, 900)
        assert preferences.get_gui_scale() == pytest.approx(stored)
        assert preferences.get_font_scale() == pytest.approx(stored)


def test_an_unreadable_stored_scale_reads_as_the_screen_default(private_store, screen):
    from spacr.qt import preferences

    store = QSettings(private_store, QSettings.IniFormat)
    store.setValue("prefs/gui_scale", "not a number")
    store.setValue("prefs/font_scale", "not a number")
    store.sync()
    screen(1440, 875)
    assert preferences.get_gui_scale() == pytest.approx(0.9)
    assert preferences.get_font_scale() == pytest.approx(0.9)


def test_an_invented_screen_uses_90_percent(private_store, monkeypatch, qapp):
    from spacr.qt import preferences

    monkeypatch.setattr(QGuiApplication, "platformName", staticmethod(lambda: "offscreen"))
    monkeypatch.setattr(QGuiApplication, "primaryScreen",
                        staticmethod(lambda: _Screen(800, 600)))
    assert preferences.get_gui_scale() == pytest.approx(0.9)
    assert preferences.get_font_scale() == pytest.approx(0.9)


def test_preferences_show_the_effective_scale_and_reset_to_it(private_store, screen, qtbot):
    from spacr.qt import preferences
    from spacr.qt.preferences import PreferencesDialog

    screen(1440, 875)
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    gui = dialog.findChild(QSlider, "GuiScale")
    font = dialog.findChild(QSlider, "FontScale")
    assert (gui.value(), font.value()) == (90, 90)

    gui.setValue(140)
    font.setValue(130)
    dialog.findChild(QPushButton, "PreferencesReset").click()
    assert (gui.value(), font.value()) == (90, 90)
    assert QSettings(private_store, QSettings.IniFormat).value("prefs/gui_scale") is None
    assert QSettings(private_store, QSettings.IniFormat).value("prefs/font_scale") is None
    assert preferences.get_gui_scale() == pytest.approx(0.9)
    assert preferences.get_font_scale() == pytest.approx(0.9)
