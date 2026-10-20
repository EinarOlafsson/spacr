"""A new user on a small screen starts at 75 % GUI and 75 % font.

The maintainer, 2026-10-20: a primary screen whose available geometry is
under about 1600 logical pixels wide -- a 13-inch 1440 x 900 MacBook -- is
small. Only a profile with no stored scale gets the smaller default; a
scale someone chose is never replaced. The screen is mocked: the offscreen
platform's own screen is invented and deliberately not measured.
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


def test_a_13_inch_laptop_starts_at_75_percent(private_store, screen):
    from spacr.qt import preferences

    screen(1440, 875)
    assert preferences.get_gui_scale() == pytest.approx(0.75)
    assert preferences.get_font_scale() == pytest.approx(0.75)


def test_a_1920_wide_screen_starts_at_100_percent(private_store, screen):
    from spacr.qt import preferences

    screen(1920, 1040)
    assert preferences.get_gui_scale() == pytest.approx(1.0)
    assert preferences.get_font_scale() == pytest.approx(1.0)


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
    assert preferences.get_gui_scale() == pytest.approx(0.75)
    assert preferences.get_font_scale() == pytest.approx(0.75)


def test_an_invented_screen_keeps_100_percent(private_store, monkeypatch, qapp):
    from spacr.qt import preferences

    monkeypatch.setattr(QGuiApplication, "platformName", staticmethod(lambda: "offscreen"))
    monkeypatch.setattr(QGuiApplication, "primaryScreen",
                        staticmethod(lambda: _Screen(800, 600)))
    assert preferences.get_gui_scale() == pytest.approx(1.0)
    assert preferences.get_font_scale() == pytest.approx(1.0)


def test_preferences_show_the_effective_scale_and_reset_to_it(private_store, screen, qtbot):
    from spacr.qt import preferences
    from spacr.qt.preferences import PreferencesDialog

    screen(1440, 875)
    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    gui = dialog.findChild(QSlider, "GuiScale")
    font = dialog.findChild(QSlider, "FontScale")
    assert (gui.value(), font.value()) == (75, 75)

    gui.setValue(140)
    font.setValue(130)
    dialog.findChild(QPushButton, "PreferencesReset").click()
    assert (gui.value(), font.value()) == (75, 75)
    assert QSettings(private_store, QSettings.IniFormat).value("prefs/gui_scale") is None
    assert preferences.get_gui_scale() == pytest.approx(0.75)
