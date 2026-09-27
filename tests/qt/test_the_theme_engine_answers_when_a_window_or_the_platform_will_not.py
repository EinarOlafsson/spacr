"""The theme engine's answers when a window, a widget or the platform will
not answer the question it is asked.

The window sheet is put on and taken off by code that walks every widget in
the application, so a widget whose C++ half is gone, a stand-in with no Qt
properties, or a platform that cannot report its colour scheme must each get
a definite answer: no sheet (``False`` / ``0``), the window sheeted whole,
the dark default. The colour helpers are pinned the same way: a palette
missing a role gives the constant accent, and a palette where no accent is
readable falls back to the ink.
"""
from __future__ import annotations

import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QMenu, QWidget  # noqa: E402

from spacr.qt import preferences as prefs  # noqa: E402
from spacr.qt import theme  # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("this part will not answer")


def _no_application(monkeypatch):
    monkeypatch.setattr(theme, "QApplication", types.SimpleNamespace(
        instance=lambda: None))


def _application(monkeypatch, app):
    monkeypatch.setattr(theme, "QApplication", types.SimpleNamespace(
        instance=lambda: app))


class _Gone:
    """A widget wrapper whose C++ half has been deleted."""

    def property(self, _name):
        raise RuntimeError("Internal C++ object already deleted.")

    def setProperty(self, _name, _value):
        raise RuntimeError("Internal C++ object already deleted.")

    def styleSheet(self):
        raise RuntimeError("Internal C++ object already deleted.")


# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

def test_the_accent_caption_follows_the_active_palette_when_given_none():
    assert theme.button_accent_text() == theme.button_accent_text(
        theme.active_palette())


def test_a_palette_missing_its_page_colour_gives_the_constant_accent():
    palette = {"fg": "#ffffff", "button_accent": "#4a9eff"}

    assert theme.button_accent_text(palette) == "#4a9eff"


def test_a_palette_where_no_accent_is_readable_falls_back_to_the_ink():
    grey = "#777777"
    palette = {role: grey for role in theme.PAGE_SURFACES}
    palette.update(bg="#777777", fg="#787878", button_accent="#767676",
                   button_accent_hi="#797979")

    assert theme.button_accent_text(palette) == "#787878"


def test_the_menu_bar_is_dark_when_the_theme_cannot_be_read(monkeypatch):
    monkeypatch.setattr(prefs, "resolve_effective_theme", _boom)

    assert theme.menu_bar_background() == theme.menu_bar_background("dark")


def test_no_backdrop_is_assumed_when_the_preference_cannot_be_read(
        monkeypatch):
    monkeypatch.setattr(prefs, "get_ambient_enabled", _boom)

    assert theme._the_backdrop_is_on() is False


# ---------------------------------------------------------------------------
# The platform's colour scheme
# ---------------------------------------------------------------------------

def test_a_platform_that_cannot_report_its_scheme_reports_none():
    app = types.SimpleNamespace(styleHints=_boom)

    assert theme.system_colour_scheme(app) is None


def test_the_scheme_is_not_held_without_an_application():
    assert theme.hold_the_colour_scheme(None, "dark") is False


def test_following_the_system_on_an_older_qt_asks_for_unknown():
    asked = []
    hints = types.SimpleNamespace(setColorScheme=asked.append)
    app = types.SimpleNamespace(styleHints=lambda: hints)

    assert theme.hold_the_colour_scheme(app, None) is True
    assert asked == [Qt.ColorScheme.Unknown]


def test_a_scheme_the_platform_refuses_is_not_held():
    app = types.SimpleNamespace(styleHints=_boom)

    assert theme.hold_the_colour_scheme(app, "dark") is False


def test_no_style_is_chosen_without_an_application(monkeypatch):
    _no_application(monkeypatch)

    assert theme.use_a_style_that_honours_the_palette() == ""


def test_a_platform_without_fusion_keeps_its_own_style(monkeypatch):
    from PySide6.QtWidgets import QStyleFactory

    monkeypatch.setattr(QStyleFactory, "create",
                        staticmethod(lambda _name: None))
    kept = []
    style = types.SimpleNamespace(name=lambda: "Windows")
    app = types.SimpleNamespace(style=lambda: style, setStyle=kept.append)

    assert theme.use_a_style_that_honours_the_palette(app, environ={}) == (
        "windows")
    assert kept == []


# ---------------------------------------------------------------------------
# Sheeting windows
# ---------------------------------------------------------------------------

class _RootsRaiseRuntime:
    @property
    def stylesheet_roots(self):
        raise RuntimeError("the window is being torn down")


def test_a_menu_that_went_away_before_it_showed_is_not_sheeted():
    assert theme._sheet_the_menu_behind(lambda: None) is None
    assert theme._sheet_the_menu_behind(lambda: _RootsRaiseRuntime()) is None


def test_a_menu_moves_its_sheet_to_about_to_show_once(qapp, qtbot):
    menu = QMenu()
    qtbot.addWidget(menu)

    assert theme._sheets_itself_before_it_shows(menu) is True
    assert menu.property(theme._SHEETS_AT_ABOUT_TO_SHOW) is True
    assert theme._sheets_itself_before_it_shows(menu) is True

    gone = _Gone()
    gone.aboutToShow = types.SimpleNamespace(connect=lambda _slot: None)
    assert theme._sheets_itself_before_it_shows(gone) is False


def test_a_window_that_cannot_name_its_roots_is_sheeted_whole():
    window = types.SimpleNamespace(stylesheet_roots=_boom)

    assert theme._roots_for(window) == [window]


def test_a_gone_widget_takes_no_mark_and_no_sheet(monkeypatch, qapp):
    widget = QWidget()
    theme.mark_as_a_sheet_target(widget)
    assert widget.property(theme._SHEET_TARGET) is True
    theme.mark_as_a_sheet_target(_Gone())

    app = types.SimpleNamespace(**{theme._WINDOW_SHEET_ATTRIBUTE: "QWidget{}"})
    _application(monkeypatch, app)
    assert theme._sheet_one_window(_Gone()) is False
    assert theme._the_sheet_can_wait_for_the_show(object()) is False
    assert theme._the_sheet_is_waiting(_Gone()) is False
    assert theme.set_a_sheeted_widgets_own_rule(_Gone(), "QLabel{}") is None
    assert theme._add_to_a_windows_own_rules(_Gone(), "QLabel{}") is False
    widget.deleteLater()


def test_nothing_is_sheeted_or_read_back_without_an_application(
        monkeypatch):
    _no_application(monkeypatch)

    assert theme._sheet_one_window(object()) is False
    assert theme.window_stylesheet() is None
    assert theme.apply_stylesheet_per_window(None, "QWidget{}") == 0


class _AppWithAClassSheet:
    """``hasattr`` finds the sheet, ``delattr`` cannot take it off."""

    _spacr_window_stylesheet = "QWidget{}"

    def allWidgets(self):
        raise RuntimeError("the application is shutting down")


def test_taking_the_sheet_off_survives_an_application_mid_shutdown():
    assert theme._WINDOW_SHEET_ATTRIBUTE == "_spacr_window_stylesheet"

    assert theme._forget_window_stylesheets(_AppWithAClassSheet()) == 0

    app = types.SimpleNamespace(allWidgets=lambda: [_Gone()])
    setattr(app, theme._WINDOW_SHEET_ATTRIBUTE, "QWidget{}")
    assert theme._forget_window_stylesheets(app) == 0
    assert not hasattr(app, theme._WINDOW_SHEET_ATTRIBUTE)
