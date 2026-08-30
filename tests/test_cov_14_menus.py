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


class _Callable:
    """A ctypes function stand-in which accepts ``restype``/``argtypes``."""

    def __init__(self, callback):
        self._callback = callback
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        return self._callback(*args)


class _ObjectiveCRuntime:
    """The tiny Objective-C surface used by the bundle-name helper."""

    bundle_class = 101
    string_class = 102
    default_bundle = 201
    default_info = 202

    def __init__(self, *, bundle=default_bundle, info=default_info,
                 class_name=b"__NSCFDictionary", failed_string=None):
        self.bundle = bundle
        self.info = info
        self.class_name = class_name
        self.failed_string = failed_string
        self.assignments = []
        self.objc_getClass = _Callable(self._get_class)
        self.sel_registerName = _Callable(lambda selector: selector)
        self.object_getClassName = _Callable(lambda info: self.class_name)
        self.objc_msgSend = object()

    def _get_class(self, name):
        return {
            b"NSBundle": self.bundle_class,
            b"NSString": self.string_class,
        }[name]

    def send(self, receiver, selector, *args):
        if receiver == self.bundle_class and selector == b"mainBundle":
            return self.bundle
        if receiver == self.bundle and selector == b"infoDictionary":
            return self.info
        if (receiver == self.string_class
                and selector == b"stringWithUTF8String:"):
            text = args[0]
            return 0 if text == self.failed_string else ("NSString", text)
        if receiver == self.info and selector == b"setObject:forKey:":
            self.assignments.append(args)
            return 1
        raise AssertionError((receiver, selector, args))


def _install_objective_c_runtime(monkeypatch, runtime):
    """Make a deterministic macOS runtime without requiring a Mac."""
    import ctypes
    import ctypes.util

    from spacr.qt import menus

    monkeypatch.setattr(menus.sys, "platform", "darwin")
    monkeypatch.delenv(menus.BUNDLE_NAME_OPT_OUT, raising=False)
    monkeypatch.setattr(
        ctypes.util, "find_library", lambda name: f"/fake/{name}")
    monkeypatch.setattr(
        ctypes.cdll, "LoadLibrary",
        lambda path: runtime if path.endswith("/objc") else object())
    monkeypatch.setattr(ctypes, "cast", lambda pointer, signature: runtime.send)


def test_the_macos_bundle_name_is_written_to_a_mutable_dictionary(monkeypatch):
    """The supported unbundled-macOS shape writes exactly CFBundleName."""
    from spacr.qt import menus

    runtime = _ObjectiveCRuntime()
    _install_objective_c_runtime(monkeypatch, runtime)

    assert menus.name_the_macos_application_menu("spaCR") is True
    assert runtime.assignments == [
        (("NSString", b"spaCR"), ("NSString", b"CFBundleName"))]


@pytest.mark.parametrize(
    ("runtime", "reason"),
    [
        (_ObjectiveCRuntime(bundle=0), "there is no main bundle"),
        (_ObjectiveCRuntime(info=0), "there is no info dictionary"),
        (_ObjectiveCRuntime(class_name=b"__NSDictionaryI"),
         "the info dictionary is immutable"),
        (_ObjectiveCRuntime(failed_string=b"spaCR"),
         "the requested name could not be encoded"),
        (_ObjectiveCRuntime(failed_string=b"CFBundleName"),
         "the bundle key could not be encoded"),
    ],
)
def test_the_macos_bundle_name_is_left_alone_when_it_is_not_writable(
        monkeypatch, runtime, reason):
    """Every runtime guard exits without attempting a dictionary mutation."""
    from spacr.qt import menus

    _install_objective_c_runtime(monkeypatch, runtime)

    assert menus.name_the_macos_application_menu("spaCR") is False, reason
    assert runtime.assignments == []


def test_an_objective_c_loader_error_is_nonfatal(monkeypatch):
    """A runtime loading error follows the helper's documented safe fallback."""
    import ctypes
    import ctypes.util

    from spacr.qt import menus

    monkeypatch.setattr(menus.sys, "platform", "darwin")
    monkeypatch.delenv(menus.BUNDLE_NAME_OPT_OUT, raising=False)
    monkeypatch.setattr(ctypes.util, "find_library", lambda name: f"/{name}")

    def fail_to_load(path):
        raise OSError(f"cannot load {path}")

    monkeypatch.setattr(ctypes.cdll, "LoadLibrary", fail_to_load)
    assert menus.name_the_macos_application_menu("spaCR") is False


def test_name_the_application_updates_both_qt_names(monkeypatch, qapp):
    """The public naming helper updates Qt and invokes the macOS hook first."""
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QGuiApplication

    from spacr.qt import menus

    old_name = QCoreApplication.applicationName()
    old_organization = QCoreApplication.organizationName()
    old_display_name = QGuiApplication.applicationDisplayName()
    bundle_names = []
    monkeypatch.setattr(
        menus, "name_the_macos_application_menu",
        lambda name: bundle_names.append(name) or False)

    try:
        result = menus.name_the_application("Coverage App", "Coverage Org")
        assert bundle_names == ["Coverage App"]
        assert result == ("Coverage App", "Coverage App")
        assert QCoreApplication.organizationName() == "Coverage Org"
    finally:
        QCoreApplication.setApplicationName(old_name)
        QCoreApplication.setOrganizationName(old_organization)
        QGuiApplication.setApplicationDisplayName(old_display_name)
