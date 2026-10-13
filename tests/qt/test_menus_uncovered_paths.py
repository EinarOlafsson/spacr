"""Menu-role pinning and application naming survive a Qt that is not there.

Both helpers in :mod:`spacr.qt.menus` are called from import-time launch code
that also has to run under a bare interpreter -- the packaging smoke checks,
the CLI entry point before a GUI is chosen, and any headless import of the
launch module. Those paths import PySide6 *inside* the function precisely so
a missing binding degrades instead of raising, and an action object that
refuses ``setMenuRole`` must not stop the rest of the menu bar from being
pinned.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication          # noqa: E402
from PySide6.QtGui import QAction, QGuiApplication   # noqa: E402

from spacr.qt import menus                           # noqa: E402


def _block_imports(monkeypatch, *blocked):
    """Make ``import`` fail for ``blocked`` module names only."""
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name in blocked:
            raise ImportError(f"no module named {name!r}")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded)


class _RefusesItsRole:
    """A menu action whose ``setMenuRole`` fails, as a proxied action can."""

    def __init__(self):
        self.attempts = []

    def setMenuRole(self, role):
        self.attempts.append(role)
        raise RuntimeError("the underlying C++ action has been deleted")


def test_a_missing_qt_binding_leaves_the_action_exactly_as_it_was(
        monkeypatch, qapp):
    """Without PySide6 the role is not assigned and the action is returned."""
    action = QAction("Preferences")
    action.setMenuRole(QAction.MenuRole.AboutRole)

    _block_imports(monkeypatch, "PySide6.QtGui")

    assert menus.set_menu_role(action, "quit") is action
    assert action.menuRole() == QAction.MenuRole.AboutRole


def test_a_missing_qt_binding_reports_no_bad_role_because_it_checks_none(
        monkeypatch, qapp):
    """The role-name check lives behind the import, so it is skipped too."""
    action = QAction("Open")

    _block_imports(monkeypatch, "PySide6.QtGui")

    assert menus.set_menu_role(action, "not-a-role") is action


def test_an_action_that_refuses_its_role_is_still_returned(qapp):
    """A failing assignment is swallowed rather than raised at the caller."""
    stubborn = _RefusesItsRole()

    assert menus.set_menu_role(stubborn, "preferences") is stubborn
    assert stubborn.attempts == [QAction.MenuRole.PreferencesRole]


def test_an_action_that_refuses_its_role_does_not_halt_the_pinning(qapp):
    """Actions after the refusing one still receive their explicit role."""
    stubborn = _RefusesItsRole()
    later = QAction("Close")
    later.setMenuRole(QAction.MenuRole.QuitRole)

    menus.pin_menu_roles([stubborn, later])

    assert later.menuRole() == QAction.MenuRole.NoRole


def test_naming_the_application_without_qt_reports_the_requested_name(
        monkeypatch, qapp):
    """The headless fallback echoes the name and leaves Qt's state alone."""
    before_name = QCoreApplication.applicationName()
    before_display = QGuiApplication.applicationDisplayName()

    _block_imports(monkeypatch, "PySide6.QtCore", "PySide6.QtGui")

    assert menus.name_the_application("Headless spaCR", "Headless Org") == (
        "Headless spaCR", "Headless spaCR")
    assert QCoreApplication.applicationName() == before_name
    assert QCoreApplication.organizationName() != "Headless Org"
    assert QGuiApplication.applicationDisplayName() == before_display


def test_naming_the_application_without_qt_still_names_the_macos_menu(
        monkeypatch, qapp):
    """The bundle-name patch runs before the binding import is attempted."""
    asked = []
    monkeypatch.setattr(menus, "name_the_macos_application_menu",
                        lambda name: asked.append(name) or False)

    _block_imports(monkeypatch, "PySide6.QtCore")

    assert menus.name_the_application("Headless spaCR") == (
        "Headless spaCR", "Headless spaCR")
    assert asked == ["Headless spaCR"]
