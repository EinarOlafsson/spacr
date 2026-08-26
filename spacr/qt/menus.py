"""Assign explicit Qt menu roles, and name the macOS application menu.

On macOS, Qt may relocate actions according to text such as "Preferences" or
"Quit". Explicit roles prevent translated or renamed actions from moving to
the wrong system-menu slot. Other platforms retain their normal menu layout.

The menu those actions are relocated INTO is named after the application, so
this module also owns naming the application -- before the ``QApplication``
exists, which is the only moment early enough for macOS to read it.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Optional, Tuple


#: What the application calls itself everywhere a platform asks.
APPLICATION_NAME = "spaCR"

#: Used by ``QSettings`` for the preferences path, so it must not drift.
ORGANIZATION_NAME = "Olafsson Lab"

#: Set to ``1`` to skip the macOS bundle-name patch below without a code
#: change, should it ever misbehave on a future macOS.
BUNDLE_NAME_OPT_OUT = "SPACR_NO_MAC_APP_NAME"


def set_menu_role(action: Any, role: str = "none") -> Any:
    """Assign a macOS menu role and return ``action``.

    Parameters
    ----------
    action : PySide6.QtGui.QAction
        Action whose role is assigned.
    role : {'none', 'preferences', 'quit', 'about'}, default='none'
        Target system-menu role. ``'none'`` keeps the action in its menu.

    Returns
    -------
    PySide6.QtGui.QAction
        The input action.

    Raises
    ------
    ValueError
        If ``role`` is not supported.
    """
    try:
        from PySide6.QtGui import QAction
    except Exception:                       # pragma: no cover - headless
        return action

    roles = {
        "none": QAction.MenuRole.NoRole,
        "preferences": QAction.MenuRole.PreferencesRole,
        "quit": QAction.MenuRole.QuitRole,
        "about": QAction.MenuRole.AboutRole,
    }
    key = str(role).strip().lower()
    if key not in roles:
        raise ValueError(
            f"unknown menu role {role!r}; expected one of {sorted(roles)}")
    try:
        action.setMenuRole(roles[key])
    except Exception:                       # pragma: no cover
        pass
    return action


def pin_menu_roles(actions: Iterable[Any],
                   preferences: Optional[Any] = None,
                   quit_action: Optional[Any] = None,
                   about: Optional[Any] = None) -> None:
    """Give every action in ``actions`` an explicit role.

    Everything gets ``NoRole`` except the three named, so adding a menu item
    later cannot accidentally claim the Preferences slot by being called
    something reasonable.

    :param actions: every action on the menu bar.
    :param preferences: the action that really is Preferences, if any.
    :param quit_action: the action that really is Quit, if any.
    :param about: the action that really is About, if any.
    """
    special = {id(a): name for a, name in (
        (preferences, "preferences"), (quit_action, "quit"), (about, "about"))
        if a is not None}
    for action in actions:
        if action is None:
            continue
        set_menu_role(action, special.get(id(action), "none"))


def name_the_macos_application_menu(name: str = APPLICATION_NAME) -> bool:
    """Set the application-menu title for an unbundled macOS launch.

    macOS derives this title from ``CFBundleName`` rather than Qt's
    ``applicationName``. Packaged applications already provide the bundle
    value and are left unchanged.

    :param name: Application-menu title.
    :returns: ``True`` if the bundle value was updated; otherwise ``False``.
    """
    if sys.platform != "darwin":
        return False
    if os.environ.get(BUNDLE_NAME_OPT_OUT, "") not in ("", "0"):
        return False
    try:
        import ctypes
        import ctypes.util

        libobjc = ctypes.util.find_library("objc")
        libfoundation = ctypes.util.find_library("Foundation")
        if not libobjc or not libfoundation:
            return False
        objc = ctypes.cdll.LoadLibrary(libobjc)
        ctypes.cdll.LoadLibrary(libfoundation)

        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.object_getClassName.restype = ctypes.c_char_p
        objc.object_getClassName.argtypes = [ctypes.c_void_p]

        def send(receiver, selector, *args, argtypes=()):
            """One typed ``objc_msgSend`` call. Untyped calls corrupt the ABI."""
            fn = ctypes.cast(
                objc.objc_msgSend,
                ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.c_void_p, *argtypes))
            return fn(receiver, objc.sel_registerName(selector), *args)

        bundle = send(objc.objc_getClass(b"NSBundle"), b"mainBundle")
        if not bundle:
            return False
        info = send(bundle, b"infoDictionary")
        if not info:
            return False

        # THE ONE CHECK THAT CANNOT ITSELF FAIL. `object_getClassName` is a
        # plain C call into the runtime -- no message is sent, so no
        # Objective-C exception can be raised, and an uncaught one would
        # abort the process rather than surface here as a Python error.
        # `__NSCFDictionary` is the toll-free-bridged class CFBundle builds
        # its info dictionary as; a frozen or immutable class name means
        # this launch is not one where the key can be written.
        class_name = objc.object_getClassName(info) or b""
        if class_name != b"__NSCFDictionary":
            return False

        nsstring = objc.objc_getClass(b"NSString")
        encoded = name.encode("utf-8")

        def to_nsstring(text: bytes):
            return send(nsstring, b"stringWithUTF8String:", text,
                        argtypes=(ctypes.c_char_p,))

        value = to_nsstring(encoded)
        key = to_nsstring(b"CFBundleName")
        if not value or not key:
            return False
        send(info, b"setObject:forKey:", value, key,
             argtypes=(ctypes.c_void_p, ctypes.c_void_p))
        return True
    except Exception:                                    # noqa: BLE001
        return False


def name_the_application(name: str = APPLICATION_NAME,
                         organization: str = ORGANIZATION_NAME
                         ) -> Tuple[str, str]:
    """Configure the Qt application and organization names.

    Call this function before constructing ``QApplication`` so the macOS
    application menu and Qt window-title fallbacks receive the configured
    name.

    :param name: Application and display name.
    :param organization: Organization name used by ``QSettings``.
    :returns: Effective ``(applicationName, applicationDisplayName)`` values.
    """
    name_the_macos_application_menu(name)
    try:
        from PySide6.QtCore import QCoreApplication
        from PySide6.QtGui import QGuiApplication
    except Exception:                       # pragma: no cover - headless
        return (name, name)
    QCoreApplication.setApplicationName(name)
    QCoreApplication.setOrganizationName(organization)
    QGuiApplication.setApplicationDisplayName(name)
    return (QCoreApplication.applicationName(),
            QGuiApplication.applicationDisplayName())
