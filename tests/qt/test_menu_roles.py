"""Every menu-bar action must have an explicit macOS menu role.

macOS moves individual actions between menus based on ``QAction::menuRole``,
and Qt assigns that role by pattern-matching the action's TEXT: anything
containing ``about``, ``config``, ``options``, ``setup``, ``settings``,
``preferences``, ``quit`` or ``exit`` is claimed, removed from the menu it
was added to, and relocated to the application menu.

That produced a bug report that read like two separate ones:

* Preferences and Quit "did not show up" in spaCR's menu -- they had been
  moved to the ``python`` menu (``python`` because spaCR runs under the
  interpreter, not as a bundled ``.app``);
* the Preferences item in that menu opened the module-recipes window,
  because ``recipes.MENU_ACTION_TEXT`` is ``"Settings recipes…"`` and the
  word *settings* was enough for Qt to give it ``PreferencesRole`` too.

None of this is observable on Linux -- Qt applies menu roles only on macOS --
so these tests assert what is platform-independent and what was actually
wrong: that no action is left for Qt to guess about. Where an item lands
still has to be confirmed on a Mac.

The failure mode being guarded is not "someone wrote the wrong role". It is
"someone renamed a menu item", which is not a change anybody expects to move
it to a different menu on one platform.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QAction                   # noqa: E402


#: The words Qt matches on. Kept here so the test that proves the heuristic
#: is real does not depend on remembering them.
_TRIGGER_WORDS = ("about", "config", "options", "setup", "settings",
                  "preferences", "quit", "exit")


@pytest.fixture
def window(qtbot, qt_theme_applied):
    """A main window with every late menu contributor installed.

    `recipes`, `walkthrough` and `feature_dictionary` all add to Help from
    outside `_build_menu_bar`, and they are exactly the ones a sweep done
    during construction would miss.
    """
    from spacr.qt.app import MainWindow

    win = MainWindow()
    qtbot.addWidget(win)
    from spacr.qt import shortcuts
    shortcuts.install(win)
    return win


def test_no_action_is_left_for_qt_to_guess_about(window):
    guessed = [a.text() for a in window._menu_bar_actions()
               if a.menuRole() == QAction.MenuRole.TextHeuristicRole]
    assert guessed == [], (
        "these have no explicit menuRole, so on macOS Qt decides from their "
        f"text and may move them to the application menu: {guessed}")


def test_exactly_three_actions_claim_a_special_role(window):
    special = {a.text(): a.menuRole() for a in window._menu_bar_actions()
               if a.menuRole() != QAction.MenuRole.NoRole}
    assert special == {
        "Preferences…": QAction.MenuRole.PreferencesRole,
        "Quit": QAction.MenuRole.QuitRole,
        "About spaCR": QAction.MenuRole.AboutRole,
    }


def test_the_recipes_action_does_not_claim_preferences(window):
    """The specific regression. Its text contains "settings"."""
    from spacr.qt.recipes import MENU_ACTION_TEXT

    matches = [a for a in window._menu_bar_actions()
               if a.text() == MENU_ACTION_TEXT]
    assert matches, f"{MENU_ACTION_TEXT!r} is not on the menu bar any more"
    for action in matches:
        assert action.menuRole() == QAction.MenuRole.NoRole


def test_the_heuristic_this_defends_against_is_real(qtbot):
    """The control.

    If Qt ever stopped defaulting to TextHeuristicRole, every assertion above
    would pass for the wrong reason -- so prove that an untouched action with
    a trigger word really is left in the guessing state.
    """
    from PySide6.QtWidgets import QWidget

    host = QWidget()
    qtbot.addWidget(host)
    naive = QAction("Settings recipes…", host)
    assert any(w in naive.text().lower() for w in _TRIGGER_WORDS)
    assert naive.menuRole() == QAction.MenuRole.TextHeuristicRole

    from spacr.qt.menus import set_menu_role
    set_menu_role(naive, "none")
    assert naive.menuRole() == QAction.MenuRole.NoRole


def test_home_preferences_and_quit_come_first(window):
    """Asked for explicitly. On macOS both are relocated to the application
    menu whatever their position, so this is what Linux and Windows show."""
    from PySide6.QtWidgets import QMenu

    bar = window.menuBar()
    spacr_menu = next(m for m in bar.findChildren(QMenu)
                      if m.title().replace("&", "") == "spaCR")
    texts = [a.text() for a in spacr_menu.actions() if not a.isSeparator()]
    # Home first, then the two Qt relocates on macOS. Asked for in that
    # order: the thing you reach for most often should not sit below
    # thirty app names. "All apps" was removed from the menu at the same
    # time -- its Ctrl+B shortcut is still registered on the window, and
    # `test_ctrl_b_survives_leaving_the_menu` covers that.
    assert texts[:3] == ["Home", "Preferences…", "Quit"], texts[:5]
    assert "All apps" not in texts, (
        "the drawer toggle is back in the menu; it was removed because the "
        "name does not say what it does")


def test_an_unknown_role_name_is_refused():
    """A typo would silently restore the guessing this exists to stop."""
    from spacr.qt.menus import set_menu_role

    host = QAction("x")
    with pytest.raises(ValueError, match="unknown menu role"):
        set_menu_role(host, "preference")      # missing the s


def test_ctrl_b_survives_leaving_the_menu(qtbot):
    """"All apps" left the menu; Ctrl+B must not have left with it.

    The edge drawer is otherwise reachable only by hovering a 6 px strip,
    which is not a route a keyboard user has. Deleting the QAction would
    have been the obvious way to remove the menu entry and would have taken
    the shortcut too.
    """
    from PySide6.QtGui import QKeySequence

    from spacr.qt.app import MainWindow

    window = MainWindow()
    qtbot.addWidget(window)

    bound = [a for a in window.actions()
             if a.shortcut() == QKeySequence("Ctrl+B")]
    assert bound, "Ctrl+B is no longer registered on the window"
    assert bound[0].text() == "All apps"
