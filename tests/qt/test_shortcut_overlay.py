"""The ``?`` overlay, and a command palette that reaches settings.

Pressing ``?`` asks "what can I press here" and the answer is worth about
two seconds. A modal dialog makes the user commit to a mode, find a close
button and leave it; an overlay answers and gets out of the way on the next
keystroke. Both halves are asserted against rendered geometry, because "the
card is centred and fits" is the whole difference between an overlay and a
rectangle of text over the middle of somebody's work.
"""
from __future__ import annotations

import pytest

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QLabel, QWidget

from spacr.qt.shortcuts import (
    SHORTCUTS,
    ShortcutOverlay,
    install,
    show_cheat_sheet,
)


@pytest.fixture
def window(qtbot):
    from spacr.qt.app import MainWindow
    win = MainWindow()
    qtbot.addWidget(win)
    win.resize(1280, 860)
    win.show()
    qtbot.waitExposed(win)
    return win


# ---------------------------------------------------------------------------
# 1. The overlay
# ---------------------------------------------------------------------------

def test_the_question_mark_gives_an_overlay_not_a_dialog(window, qtbot):
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    assert isinstance(overlay, ShortcutOverlay)
    assert overlay.parentWidget() is window
    overlay.dismiss()


def test_the_overlay_covers_the_window(window, qtbot):
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    assert overlay.size() == window.rect().size()
    overlay.dismiss()


def test_the_card_is_centred_and_fits(window, qtbot):
    """Rendered geometry, not a promise about a layout."""
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    overlay.show()
    qtbot.waitExposed(overlay)

    card = overlay._card
    assert card.width() > 0 and card.height() > 0
    assert card.width() <= overlay.width()
    assert card.height() <= overlay.height()

    card_centre = card.geometry().center()
    overlay_centre = overlay.rect().center()
    assert abs(card_centre.x() - overlay_centre.x()) <= 1
    assert abs(card_centre.y() - overlay_centre.y()) <= 1
    overlay.dismiss()


def test_every_registered_shortcut_is_on_the_card(window, qtbot):
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    rendered = {label.text() for label in overlay._card.findChildren(QLabel)}
    for spec in SHORTCUTS:
        assert spec.keys in rendered, f"{spec.keys} is bound but not shown"
        assert spec.label in rendered, f"{spec.label} is missing its row"
    overlay.dismiss()


def test_the_categories_are_laid_out_in_columns(window, qtbot):
    """Fifteen bindings in one column is a scroll; in three it is a glance."""
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    overlay.show()
    qtbot.waitExposed(overlay)
    headers = [lbl for lbl in overlay._card.findChildren(QLabel)
               if lbl.objectName() == "ShortcutOverlayCategory"]
    assert len(headers) >= 2
    xs = {lbl.x() for lbl in headers}
    assert len(xs) == len(headers), "the categories stacked instead of tiling"
    overlay.dismiss()


def test_any_key_closes_it(window, qtbot):
    overlay = show_cheat_sheet(window)
    qtbot.addWidget(overlay)
    overlay.keyPressEvent(
        QKeyEvent(QKeyEvent.KeyPress, Qt.Key_A, Qt.NoModifier, "a"))
    assert not overlay.isVisible()


def test_asking_twice_leaves_one_overlay(window, qtbot):
    first = show_cheat_sheet(window)
    qtbot.addWidget(first)
    second = show_cheat_sheet(window)
    qtbot.addWidget(second)
    assert second is not first
    assert not first.isVisible()
    live = [o for o in window.findChildren(ShortcutOverlay) if o.isVisible()]
    assert len(live) == 1
    second.dismiss()


def test_a_sizeless_parent_still_gets_the_dialog(qtbot, monkeypatch):
    """The fallback exists for a parentless or zero-sized caller, where an
    overlay would have nothing to cover."""
    bare = QWidget()
    qtbot.addWidget(bare)
    bare.resize(0, 0)
    shown = []
    monkeypatch.setattr(
        "PySide6.QtWidgets.QDialog.exec", lambda self: shown.append(self))
    assert show_cheat_sheet(bare) is None
    assert shown


def test_the_new_bindings_are_both_declared_and_wired(window):
    """A binding documented but not bound, or bound but not documented, is
    the failure `SHORTCUTS` exists to prevent."""
    declared = {s.keys for s in SHORTCUTS}
    assert {"Ctrl+F", "Ctrl+Shift+R"} <= declared

    from PySide6.QtGui import QShortcut
    bound = {sc.key().toString() for sc in window.findChildren(QShortcut)}
    for keys in declared:
        assert keys in bound, f"{keys} is on the cheat sheet but not bound"


def test_ctrl_f_lands_in_the_settings_search_box(window, qtbot):
    from spacr.qt.shortcuts import _focus_settings_search
    window._on_nav_selected("mask")
    qtbot.wait(50)
    screen = window._screens.get("mask")
    bar = getattr(screen, "_settings_search", None)
    assert bar is not None, "the strip was never installed"
    _focus_settings_search(window)
    assert bar._input.hasFocus()


def test_ctrl_f_is_harmless_on_a_screen_without_a_form(window):
    from spacr.qt.shortcuts import _focus_settings_search
    window._on_nav_selected("__home__")
    _focus_settings_search(window)  # must not raise


# ---------------------------------------------------------------------------
# 2. The palette reaches settings
# ---------------------------------------------------------------------------

def _palette(window, qtbot):
    from spacr.qt.command_palette import CommandPalette
    palette = CommandPalette(window)
    qtbot.addWidget(palette)
    return palette


def test_the_palette_lists_the_current_modules_settings(window, qtbot):
    window._on_nav_selected("mask")
    qtbot.wait(50)
    palette = _palette(window, qtbot)
    settings = [c for c in palette._commands
                if c.section.startswith("Settings")]
    assert len(settings) == len(window._screens["mask"]._settings_model._widgets)


def test_a_setting_is_findable_in_the_palette_by_its_description(window,
                                                                  qtbot):
    """Same haystack as the settings strip: the description is the only part
    written in the language a user thinks in."""
    window._on_nav_selected("measure")
    qtbot.wait(50)
    palette = _palette(window, qtbot)
    palette._on_filter("straddling")
    labels = [palette._list.item(i).text()
              for i in range(palette._list.count())]
    assert any("merge_edge_pathogen_cells" in label for label in labels)


def test_activating_a_setting_reveals_it_on_the_form(window, qtbot):
    window._on_nav_selected("mask")
    qtbot.wait(50)
    screen = window._screens["mask"]
    palette = _palette(window, qtbot)
    palette._reveal_setting("merge_pathogens")

    bar = screen._settings_search
    assert bar.query() == "merge_pathogens"
    assert bar.visible_keys() == ["merge_pathogens"]
    holding = bar._index["merge_pathogens"][0]
    assert holding.is_expanded(), (
        "the palette named the setting and left it behind a closed heading")


def test_the_palette_scopes_settings_to_the_module_on_screen(window, qtbot):
    """Every setting of every module is over a thousand rows, and eleven
    identically-named `diameter` entries is a worse answer than none."""
    window._on_nav_selected("mask")
    qtbot.wait(50)
    sections = {c.section for c in _palette(window, qtbot)._commands}
    assert "Settings · mask" in sections
    assert "Settings · measure" not in sections


def test_the_palette_has_no_settings_section_on_home(window, qtbot):
    window._on_nav_selected("__home__")
    qtbot.wait(20)
    sections = {c.section for c in _palette(window, qtbot)._commands}
    assert not any(s.startswith("Settings") for s in sections)


def test_menu_commands_survive_being_collected(window, qtbot):
    """Menu actions are mirrored into the palette. Reaching them through
    ``QAction.menu()`` produced entries that raised "Internal C++ object
    already deleted" the moment they were triggered."""
    palette = _palette(window, qtbot)
    menu_cmds = [c for c in palette._commands if c.section == "Menu"]
    assert menu_cmds
    for cmd in menu_cmds:
        assert " → " in cmd.label
    labels = {c.label for c in menu_cmds}
    assert any("Settings recipes" in label for label in labels)


def test_installing_shortcuts_twice_is_harmless(window):
    install(window)  # already installed by MainWindow.__init__
    assert getattr(window, "_settings_search_watcher", None) is not None
