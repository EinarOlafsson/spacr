"""Pinning menu roles survives a gap in the action list.

Menu bars are assembled by appending, and a separator or a slot that has not
been filled yet comes through as ``None``. Pinning the roles must skip those
rather than raising, because the alternative is a half-pinned menu bar: the
actions after the gap keep whatever role Qt guessed from their text, which is
exactly the accidental Preferences/Quit relocation this module exists to
prevent.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.menus import pin_menu_roles, set_menu_role  # noqa: E402


def _action(text):
    from PySide6.QtGui import QAction

    return QAction(text)


def test_a_gap_in_the_menu_does_not_stop_the_pinning(qapp):
    """Actions after a ``None`` entry still get an explicit role."""
    from PySide6.QtGui import QAction

    first = _action("Open")
    last = _action("Close")
    prefs = _action("Settings")

    pin_menu_roles([first, None, prefs, None, last], preferences=prefs)

    assert first.menuRole() == QAction.MenuRole.NoRole
    assert last.menuRole() == QAction.MenuRole.NoRole
    assert prefs.menuRole() == QAction.MenuRole.PreferencesRole


def test_the_three_named_slots_are_the_only_ones_claimed(qapp):
    """Quit and About are pinned by identity, not by their text."""
    from PySide6.QtGui import QAction

    quit_action = _action("Leave")
    about = _action("Credits")
    decoy = _action("Quit")

    pin_menu_roles([quit_action, about, decoy],
                   quit_action=quit_action, about=about)

    assert quit_action.menuRole() == QAction.MenuRole.QuitRole
    assert about.menuRole() == QAction.MenuRole.AboutRole
    assert decoy.menuRole() == QAction.MenuRole.NoRole


def test_an_unknown_role_name_is_refused(qapp):
    """A typo in a role name is a programming error, not a silent NoRole."""
    with pytest.raises(ValueError):
        set_menu_role(_action("Open"), "preference")
