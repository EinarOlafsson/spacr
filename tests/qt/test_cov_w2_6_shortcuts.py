"""A shortcut that cannot fire must not cost the user the window.

Every route out of :mod:`spacr.qt.shortcuts` ends in an optional module --
the command palette, Preferences, recipes, a screen's own search box -- and
each of them is guarded on its own. These drive the real handlers against
real widgets where a widget is what the code touches, and inject the failure
where the point is what happens when an optional piece is missing.
"""
from __future__ import annotations

import logging

import pytest
from PySide6.QtCore import QSize
from PySide6.QtGui import QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow

from spacr.qt import shortcuts as sc


@pytest.fixture
def window(qapp):
    win = QMainWindow()
    win.resize(900, 600)
    yield win
    win.close()
    win.deleteLater()


# --------------------------------------------------------------------------
# the declared map
# --------------------------------------------------------------------------


def test_the_map_is_the_window_wide_keys_plus_the_per_screen_ones():
    """The distinction is real -- one set is always live and the other is
    not -- and the map has to describe both."""
    everything = sc.mapped()
    assert everything[:len(sc.SHORTCUTS)] == sc.SHORTCUTS
    # 656: a key shared by several screens (Left/Right) is listed once per
    # screen, because each screen now holds its own rebindable copy of it.
    annotate_navigation = {
        "Left": "Move focus left", "Right": "Move focus right",
        "PageUp": "Previous page", "Alt+Left": "Previous page",
        "PageDown": "Next page", "Alt+Right": "Next page",
    }
    assert {(spec.keys, annotate_navigation.get(spec.keys, spec.label)
             if spec.category == "Annotate" else spec.label)
            for spec in sc.SCREEN_SHORTCUTS} <= {
        (spec.keys, spec.label) for spec in everything}
    assert set(annotate_navigation.items()) <= {
        (spec.keys, spec.label) for spec in everything
        if spec.category == "Annotate"}
    for scope in ("Annotate", "Make Masks", "Field browser"):
        assert {"Left", "Right"} <= {
            spec.keys for spec in everything[len(sc.SHORTCUTS):]
            if spec.category == scope}
    assert all(spec.scope != sc.EVERYWHERE for spec in sc.SCREEN_SHORTCUTS)


# --------------------------------------------------------------------------
# discovering what is live
# --------------------------------------------------------------------------


def test_a_window_that_cannot_be_searched_discovers_nothing():
    class Hostile:
        def findChildren(self, *_args):
            raise RuntimeError("the window is already gone")

    assert sc.discover(Hostile()) == []


def test_a_holder_that_cannot_name_its_key_is_skipped_not_fatal(window):
    """One unreadable action must not empty the whole cheat sheet."""
    class Broken:
        def shortcut(self):
            raise RuntimeError("deleted underneath us")

    class Mixed:
        def findChildren(self, kind):
            return [Broken()] if kind is not QShortcut else []

    assert sc.discover(Mixed()) == []


# --------------------------------------------------------------------------
# what each key actually does
# --------------------------------------------------------------------------


def test_a_navigation_key_on_a_window_without_the_route_is_ignored():
    """The missing route has to be the reason nothing happened, so the
    lookup is recorded: a guard that bailed earlier would never have asked
    for `_on_nav_selected` at all."""
    asked = []

    class NoRoute:
        def __getattr__(self, name):
            asked.append(name)
            raise AttributeError(name)

    sc._nav(NoRoute(), "__home__")
    assert asked == ["_on_nav_selected"]


def test_the_palette_key_opens_the_palette(monkeypatch, window):
    from spacr.qt import command_palette

    opened = []

    class FakePalette:
        def __init__(self, parent):
            opened.append(parent)

        def exec(self):
            opened.append("exec")
            return 0

        def deleteLater(self):
            opened.append("released")

    monkeypatch.setattr(command_palette, "CommandPalette", FakePalette)
    sc._open_palette(window)
    assert opened == [window, "exec", "released"]


def test_a_palette_that_cannot_be_built_is_not_a_crash(monkeypatch, window,
                                                       caplog):
    """The palette was really reached -- a guard that never tried also does
    not raise -- and what went wrong is left in the log rather than lost."""
    from spacr.qt import command_palette

    tried = []

    def _explode(parent):
        tried.append(parent)
        raise RuntimeError("no registry")

    monkeypatch.setattr(command_palette, "CommandPalette", _explode)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.shortcuts"):
        sc._open_palette(window)
    assert tried == [window]
    assert "no registry" in caplog.text


def test_the_preferences_key_opens_preferences(monkeypatch, window):
    from spacr.qt import preferences

    opened = []

    class FakeDialog:
        def __init__(self, parent):
            opened.append(parent)

        def exec(self):
            return 0

    monkeypatch.setattr(preferences, "PreferencesDialog", FakeDialog)
    sc._open_preferences(window)
    assert opened == [window]


def test_preferences_that_cannot_be_built_is_not_a_crash(monkeypatch, window,
                                                        caplog):
    """As with the palette: the dialog was attempted, and the reason it
    could not be built survives in the log."""
    from spacr.qt import preferences

    tried = []

    def _explode(parent):
        tried.append(parent)
        raise RuntimeError("no settings store")

    monkeypatch.setattr(preferences, "PreferencesDialog", _explode)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.shortcuts"):
        sc._open_preferences(window)
    assert tried == [window]
    assert "no settings store" in caplog.text


def test_the_recipes_key_opens_the_recipe_menu(monkeypatch, window):
    from spacr.qt import recipes

    fired = []

    class FakeHandler:
        def __init__(self, parent):
            self.parent = parent

        def on_triggered(self):
            fired.append(self.parent)

    monkeypatch.setattr(recipes, "_RecipeMenuHandler", FakeHandler)
    sc._open_recipes(window)
    assert fired == [window]


def test_a_recipe_dialog_that_is_unavailable_is_not_a_crash(monkeypatch,
                                                            window, caplog):
    """The handler was constructed against this window before it refused,
    and the refusal is logged rather than silently dropped."""
    from spacr.qt import recipes

    tried = []

    def _explode(parent):
        tried.append(parent)
        raise RuntimeError("no module on screen")

    monkeypatch.setattr(recipes, "_RecipeMenuHandler", _explode)
    with caplog.at_level(logging.DEBUG, logger="spacr.qt.shortcuts"):
        sc._open_recipes(window)
    assert tried == [window]
    assert "no module on screen" in caplog.text


# --------------------------------------------------------------------------
# the overlay
# --------------------------------------------------------------------------


@pytest.mark.parametrize('font_scale', [.9, 2.])
def test_parentless_shortcut_dialog_fits_scrolls_and_remains_resizable(
        monkeypatch, qtbot, qt_theme_applied, font_scale):
    from PySide6.QtCore import QPoint
    from PySide6.QtWidgets import QScrollArea, QSizeGrip
    from spacr.qt import preferences
    from spacr.qt.gui_scale import _resize_actual
    from spacr.qt.widgets.workflow_diagram import DiagramDialog

    previous = preferences.get_font_scale()
    preferences.set_font_scale(font_scale)
    captured = []

    def inspect(dialog):
        captured.append(dialog)
        qtbot.addWidget(dialog)
        dialog.show()
        qtbot.waitExposed(dialog)
        scroll = dialog.findChild(QScrollArea)
        assert scroll is not None
        assert dialog.screen().availableGeometry().contains(dialog.windowHandle().frameGeometry())
        labels = scroll.widget().findChildren(QLabel)
        rows = [label for label in labels if label.text().startswith('<code')]
        assert len(rows) == len(sc.SHORTCUTS)
        expected = [spec.label for category in dict.fromkeys(spec.category for spec in sc.SHORTCUTS)
                    for spec in sc.SHORTCUTS if spec.category == category]
        assert [label.text().split('&nbsp; ', 1)[-1] for label in rows] == expected
        assert dialog.findChildren(QSizeGrip)
        assert dialog.maximumSize().height() > dialog.screen().availableGeometry().height()
        initial = dialog.windowHandle().geometry().size()
        grown = QSize(initial.width() + 10, initial.height() + 10)
        _resize_actual(dialog, grown)
        qtbot.waitUntil(lambda: dialog.windowHandle().geometry().size() == grown)
        target = QSize(initial.width(), max(180, initial.height() // 2))
        _resize_actual(dialog, target)
        qtbot.waitUntil(lambda: dialog.windowHandle().geometry().height() <= target.height())
        assert scroll.verticalScrollBar().maximum() > 0
        last = rows[-1]
        scroll.ensureWidgetVisible(last)
        qtbot.waitUntil(lambda: scroll.viewport().rect().contains(
            last.rect().translated(last.mapTo(scroll.viewport(), QPoint()))))
        return 0

    monkeypatch.setattr(DiagramDialog, 'exec', inspect)
    try:
        assert sc.show_cheat_sheet(None) is None
        assert len(captured) == 1
    finally:
        preferences.set_font_scale(previous)


def test_the_overlay_qss_names_the_card_the_theme_has_to_reach():
    palette = {"theme": "dark", "accent": "#00aaff", "fg": "#eeeeee",
               "fg_dim": "#999999"}
    qss = sc._overlay_qss(palette, 1.0)
    assert f"QWidget#{sc.OVERLAY_CARD_NAME}" in qss
    assert "#00aaff" in qss
