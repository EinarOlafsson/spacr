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


# ---------------------------------------------------------------------------
# a tutorial step that points at a fold switch
# ---------------------------------------------------------------------------

def test_a_step_pointing_at_a_screen_that_is_not_there_highlights_nothing():
    """The tutorial runs against whatever screen is open.

    A step written for a screen the user has since closed must resolve to
    None rather than raise -- the tutorial would otherwise stop dead at a
    step that is merely no longer applicable.
    """
    from spacr.qt.tutorial.scripts import _fold_button

    assert _fold_button(None, "timelapse") is None


def test_a_screen_with_no_fold_strip_says_so_in_the_log(caplog):
    """A step that highlights nothing is a step the user cannot follow.

    Returning None silently would leave the tutorial pointing at empty space
    with no record of why, so the warning names the key it could not find.
    """
    import logging

    from spacr.qt.tutorial.scripts import _fold_button

    class _NoStrip:
        _fold_strip = None

    with caplog.at_level(logging.WARNING):
        assert _fold_button(_NoStrip(), "timelapse") is None

    assert "carries no fold strip" in caplog.text
    assert "timelapse" in caplog.text


def test_a_strip_that_is_not_a_strip_is_refused_by_shape(caplog):
    """``button_for`` is the whole interface, checked rather than assumed.

    A screen can carry an attribute of that name that is not a fold strip --
    a layout, a placeholder set during a rebuild -- and calling it would
    raise inside the tutorial rather than skipping a step.
    """
    import logging

    from spacr.qt.tutorial.scripts import _fold_button

    class _NotAStrip:
        _fold_strip = object()

    with caplog.at_level(logging.WARNING):
        assert _fold_button(_NotAStrip(), "motility") is None

    assert "carries no fold strip" in caplog.text


def test_a_strip_with_no_switch_for_that_key_names_the_key(caplog):
    """The strip exists and simply has no such switch on this screen.

    A different warning from the one above, because the fix is different: one
    is a screen without the strip, the other is a step naming a fold this
    screen does not offer.
    """
    import logging

    from spacr.qt.tutorial.scripts import _fold_button

    class _Strip:
        def button_for(self, key):
            return None

    class _Screen:
        _fold_strip = _Strip()

    with caplog.at_level(logging.WARNING):
        assert _fold_button(_Screen(), "a_fold_that_does_not_exist") is None

    assert "no fold switch for" in caplog.text
    assert "a_fold_that_does_not_exist" in caplog.text


def test_a_strip_that_has_the_switch_hands_it_back():
    """Otherwise the four refusals above would pass on a constant None."""
    from spacr.qt.tutorial.scripts import _fold_button

    sentinel = object()

    class _Strip:
        def button_for(self, key):
            return sentinel if key == "timelapse" else None

    class _Screen:
        _fold_strip = _Strip()

    assert _fold_button(_Screen(), "timelapse") is sentinel
