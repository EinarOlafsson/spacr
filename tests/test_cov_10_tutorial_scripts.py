"""Where a tutorial's cursor aims when the menu it names has moved.

A tutorial video is rendered from a script that names menus by title. The
branches here are the ones that keep the cursor pointing at something real:
a menu that sits on the bar aims at its own title, a menu that has been moved
under another one aims at the parent a user actually clicks, and a lookup
that cannot be performed at all returns nothing rather than aiming at (0, 0)
-- the corner of the screen, where a recorded click means nothing.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QMainWindow, QMenu       # noqa: E402

from spacr.qt.tutorial import scripts as S             # noqa: E402


@pytest.fixture
def window(qtbot):
    win = QMainWindow()
    qtbot.addWidget(win)
    bar = win.menuBar()
    file_menu = bar.addMenu("File")
    file_menu.addAction("Open")
    help_menu = bar.addMenu("Help")
    demos = QMenu("Demos", help_menu)
    help_menu.addMenu(demos)
    bar.addSeparator()
    bar.resize(400, 24)
    return win


def test_a_menu_on_the_bar_is_aimed_at_its_own_title(window):
    """The cursor lands on the words the user reads. Aiming at the bar's
    centre instead would put the click on whatever menu happens to sit in the
    middle."""
    bar, point = S._menu_target(window, "File")
    assert bar is window.menuBar()
    rect = bar.actionGeometry(S._find_menu(window, "File").menuAction())
    assert point == (rect.center().x(), rect.center().y())


def test_a_submenu_is_aimed_at_the_top_level_menu_that_opens_it(window):
    """A menu moved under another one has no geometry of its own on the bar,
    so its own rectangle is empty and the cursor would aim at (0, 0). It has
    to point at the menu a user's hand actually goes to."""
    bar, point = S._menu_target(window, "Demos")
    help_rect = bar.actionGeometry(S._find_menu(window, "Help").menuAction())
    assert point == (help_rect.center().x(), help_rect.center().y())


def test_a_bar_entry_that_opens_no_menu_is_skipped(window):
    """A separator, or a plain action on the bar, has no submenus to search.
    Treating it as a menu would raise while looking for the parent of one."""
    assert S._top_level_menu_containing(
        window, S._find_menu(window, "Demos")).title() == "Help"
    assert S._top_level_menu_containing(
        window, S._find_menu(window, "File")) is None


def test_a_menu_lookup_that_cannot_run_finds_nothing(window, monkeypatch):
    """The lookup lives in another module. If it cannot be imported the
    script must fall back to "no menu here" rather than taking the whole
    tutorial render down."""
    monkeypatch.setitem(sys.modules, "spacr.qt.first_run",
                        types.ModuleType("spacr.qt.first_run"))
    assert S._find_menu(window, "File") is None
    bar, point = S._menu_target(window, "File")
    assert bar is window.menuBar()
    assert point is None
