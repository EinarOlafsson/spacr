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
    """Make macOS title the application menu ``name``. Returns whether it ran.

    The menu beside the Apple logo -- the one macOS moves Preferences, Quit
    and About into -- is titled from the running bundle's ``CFBundleName``,
    NOT from :func:`QCoreApplication.applicationName`. A launch that is not a
    packaged ``.app`` has no bundle of its own and inherits the interpreter's,
    so the menu reads ``Python`` (or the console script's name) and a user
    looking for Preferences under an application called spaCR does not find
    it.

    Qt reads that key through ``CFBundleGetValueForInfoDictionaryKey``, which
    returns whatever the bundle's info dictionary currently holds -- so
    writing the key before the ``QApplication`` is constructed is enough. A
    real ``.app`` already ships the right ``CFBundleName`` and is left alone.

    :param name: the name the application menu should carry.
    :returns: ``True`` when the key was written, ``False`` on every platform
        and every launch where it was not -- never an exception, because a
        cosmetic menu title is not worth a failed start.
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
    """Name the application to Qt and to macOS, and return what took effect.

    CALL THIS BEFORE CONSTRUCTING THE ``QApplication``. Both names are static
    on ``QCoreApplication``/``QGuiApplication`` and survive construction, and
    macOS builds its application menu -- "About spaCR", "Preferences…",
    "Hide spaCR", "Quit spaCR" -- while the Cocoa platform plugin comes up
    inside that constructor. A name assigned afterwards is a name that menu
    never saw. Left unset, the name is whatever ``argv[0]`` happened to be:
    the console script, the script file, or ``PySideApp`` when the argument
    list is empty.

    ``applicationDisplayName`` is set as well as ``applicationName``. It is
    the one Qt shows to people -- window titles fall back to it -- and unlike
    the other it has no default beyond mirroring ``applicationName``.

    :param name: the application's name.
    :param organization: the organization name ``QSettings`` keys on.
    :returns: ``(applicationName, applicationDisplayName)`` as read back, so
        a caller can assert what actually took rather than assume.
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
