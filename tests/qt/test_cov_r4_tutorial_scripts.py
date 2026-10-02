"""Where a tutorial aims when the menu it names is not where it was.

A recorded tutorial names menus by title and hands the engine a point to
move the cursor to. Pins the paths taken when that lookup goes wrong: a
deleted menu wrapper, a detached submenu, an empty bar, a hidden parent entry
with no geometry, and a fold switch the screen does not carry. Every one of
them has to answer "nowhere" rather than (0, 0) -- the corner of the screen,
where a recorded click means nothing.
"""
from __future__ import annotations

import logging
import types

import pytest

from PySide6.QtWidgets import QMainWindow, QMenu

from spacr.qt.tutorial import scripts as S


@pytest.fixture
def window(qtbot):
    """A bar carrying File, Help > Reference, and a plain action beside them."""
    win = QMainWindow()
    qtbot.addWidget(win)
    bar = win.menuBar()
    file_menu = bar.addMenu("File")
    file_menu.addAction("Open")
    help_menu = bar.addMenu("Help")
    reference = QMenu("Reference", help_menu)
    help_menu.addMenu(reference)
    win._reference_menu = reference
    # A bar entry that is an action rather than a menu: the submenu scan has
    # to step over it instead of asking it for its entries.
    bar.addAction("Quit")
    bar.resize(400, 24)
    return win


def _deleted(menu):
    """Drop ``menu``'s C++ object, leaving the Python wrapper behind."""
    import shiboken6

    shiboken6.delete(menu)
    return menu


# --------------------------------------------------------------------------
# finding the menu
# --------------------------------------------------------------------------

def test_a_deleted_unattached_wrapper_does_not_replace_the_bar_menu(window):
    """Menu lookup follows bar ownership instead of unrelated saved wrappers."""
    live = S._find_menu(window, "Reference")
    assert live is window._reference_menu
    window._reference_menu = _deleted(QMenu("Reference", window))
    found = S._find_menu(window, "Reference")
    assert found is live
    assert found is not window._reference_menu


def test_a_detached_menu_has_no_target_even_when_help_exists(qtbot, caplog):
    """An unrelated Help action must not stand in for an absent destination."""
    win = QMainWindow()
    qtbot.addWidget(win)
    bar = win.menuBar()
    bar.addMenu("File").addAction("Open")
    bar.addMenu("Help")
    bar.addAction("Quit")
    bar.resize(400, 24)
    loose = QMenu("Reference", win)
    assert S._find_menu(win, "Reference") is None
    assert S._top_level_menu_containing(win, loose) is None
    with caplog.at_level(logging.WARNING, logger="spacr.qt.tutorial"):
        _bar, point = S._menu_target(win, "Reference")
    assert point is None
    assert "has no 'Reference' menu" in caplog.text


def test_a_bar_with_nothing_on_it_aims_the_cursor_nowhere(qtbot, caplog):
    """An empty menu bar has no actionable recording target."""
    win = QMainWindow()
    qtbot.addWidget(win)
    bar = win.menuBar()
    with caplog.at_level(logging.WARNING, logger="spacr.qt.tutorial"):
        _bar, point = S._menu_target(win, "File")
    assert point is None
    assert "has no 'File' menu" in caplog.text
    file_menu = bar.addMenu("File")
    bar.resize(400, 24)
    _bar, point = S._menu_target(win, "File")
    rect = bar.actionGeometry(file_menu.menuAction())
    assert point == (rect.center().x(), rect.center().y())


def test_a_parent_entry_with_no_rectangle_aims_the_cursor_nowhere(window, caplog):
    """A hidden submenu parent must not become a click at the bar's origin."""
    bar = window.menuBar()
    help_menu = S._find_menu(window, "Help")
    help_menu.menuAction().setVisible(False)
    assert not bar.actionGeometry(help_menu.menuAction()).width()
    with caplog.at_level(logging.WARNING, logger="spacr.qt.tutorial"):
        _bar, point = S._menu_target(window, "Reference")
    assert point is None
    assert "has no 'Reference' menu" in caplog.text
    help_menu.menuAction().setVisible(True)
    _bar, point = S._menu_target(window, "Reference")
    rect = bar.actionGeometry(help_menu.menuAction())
    assert point == (rect.center().x(), rect.center().y())


# --------------------------------------------------------------------------
# which bar menu a submenu hangs under
# --------------------------------------------------------------------------

def test_a_menu_whose_object_has_gone_belongs_to_no_top_level_menu(window):
    """Asked of a wrapper Qt has already released. Reading its geometry
    raises, and a relation that cannot be read is not a relation."""
    assert S._top_level_menu_containing(window, window._reference_menu) is not None
    assert S._top_level_menu_containing(window, _deleted(QMenu("Reference"))) is None


def test_a_menu_that_cannot_say_its_title_belongs_to_no_top_level_menu(window):
    """The scan compares semantic titles, so a wrapper that cannot produce
    one has nothing to compare -- and title-only relation scans must never
    guess."""
    loose = QMenu("Loose", window)

    class _NoTitle:
        """Live enough to be measured on the bar, dead enough to be read."""

        def menuAction(self):
            return loose.menuAction()

        def title(self):
            raise RuntimeError("wrapped C++ object already deleted")

    assert S._top_level_menu_containing(window, _NoTitle()) is None
    assert S._top_level_menu_containing(window, window._reference_menu) is not None


def test_an_entry_that_has_gone_does_not_stop_the_submenu_scan(window,
                                                               monkeypatch):
    """One released entry on one menu must not cost the tutorial the whole
    relation: the scan carries on to the entry that does answer."""
    real_find = S._find_menu
    help_menu = real_find(window, "Help")

    class _DeadEntry:
        def menu(self):
            raise RuntimeError("wrapped C++ object already deleted")

    class _WithADeadEntry:
        """``Help``, with a released entry in front of Reference."""

        inner = help_menu

        def actions(self):
            return [_DeadEntry(), *help_menu.actions()]

    def _patched(win, title):
        found = real_find(win, title)
        return _WithADeadEntry() if found is help_menu else found

    monkeypatch.setattr(S, "_find_menu", _patched)
    parent = S._top_level_menu_containing(window, window._reference_menu)
    assert isinstance(parent, _WithADeadEntry)
    assert parent.inner is help_menu


# --------------------------------------------------------------------------
# the fold switch
# --------------------------------------------------------------------------

def test_a_screen_with_no_fold_strip_highlights_nothing_and_says_so(caplog):
    """A fold button IS the module's icon and carries no text, so a text
    search cannot find it -- the strip is the only way to ask. A screen
    without one has to be reported, because the step will otherwise
    highlight nothing and nobody will know why."""
    strip = types.SimpleNamespace(button_for=lambda key: f"switch:{key}")
    screen = types.SimpleNamespace(_fold_strip=strip)
    assert S._fold_button(screen, "mask") == "switch:mask"

    with caplog.at_level(logging.WARNING, logger="spacr.qt.tutorial"):
        assert S._fold_button(types.SimpleNamespace(), "mask") is None
    assert "carries no fold strip" in caplog.text


def test_a_strip_without_a_switch_for_that_module_says_so(caplog):
    """The strip is there and the module is not on it. Silently returning
    nothing would leave a step pointing at the last thing highlighted."""
    strip = types.SimpleNamespace(
        button_for=lambda key: "switch" if key == "mask" else None)
    screen = types.SimpleNamespace(_fold_strip=strip)
    assert S._fold_button(screen, "mask") == "switch"

    with caplog.at_level(logging.WARNING, logger="spacr.qt.tutorial"):
        assert S._fold_button(screen, "measure") is None
    assert "no fold switch for 'measure'" in caplog.text
