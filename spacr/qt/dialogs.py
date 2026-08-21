"""Window behaviour for spaCR's modal dialogs.

One function, and the reason it exists is worth more than the code.

A ``QDialog`` created with a parent is *transient-for* that parent, and
``exec()`` makes it modal. A window manager that implements **attached modal
dialogs** -- GNOME/Mutter does by default, and it is not alone -- treats that
combination as an invitation to glue the dialog to its parent: it is drawn
centred on the parent, it cannot be dragged anywhere else, and pulling at it
manipulates the parent window instead. On a maximised main window that reads
as "the app un-maximised itself when I tried to move the settings".

The tell is which dialogs misbehave. In spaCR, Preferences
(``PreferencesDialog(...).exec()``) and Annotate's settings
(``_SettingsDialog(...).exec()``) are attached; the UMAP search settings
(``UmapSearchSettingsDialog(panel).show()``) and the Mask and crop live
preview settings are not. The first two are modal and the rest are modeless
-- which is exactly the WM's rule, and nothing to do with how any of them
are built.

The fix is to stop advertising them as dialogs. Clearing ``Qt.Dialog`` from
the window type and setting ``Qt.Window`` makes the WM see an ordinary
top-level window with its own frame, which nothing attaches. The parent is
kept, so the dialog still stacks above the app and is still owned by it, and
the modality is untouched -- ``exec()`` still blocks and still returns
``Accepted``.

Not done instead:

* **Dropping the parent.** It detaches, but the dialog then stops stacking
  above the main window and can be lost behind it, and Qt no longer destroys
  it with its owner.
* **Making the dialogs modeless.** Every caller reads the ``exec()`` result
  to decide whether to apply the settings. Changing that is a different and
  much larger piece of work.
* **A platform check.** The flags are harmless on Windows and macOS -- a
  dialog with a normal window type is a normal window there too -- and a
  conditional would mean the layout differs by platform for no gain.
"""
from __future__ import annotations

from typing import Any


def detach_from_window_manager(dialog: Any) -> Any:
    """Stop a WM attaching ``dialog`` to its parent. Returns ``dialog``.

    Call it before ``exec()`` on any modal dialog the user should be able to
    drag where they like. Safe to call more than once, and safe on a dialog
    with no parent.

    :param dialog: the ``QDialog`` (or any ``QWidget`` shown as a window).
    :returns: the same object, so it can be used inline.
    """
    try:
        from PySide6.QtCore import Qt
    except Exception:                       # pragma: no cover - headless import
        return dialog
    try:
        flags = dialog.windowFlags()
        # `Qt.Dialog` is `Qt.Window | 0x2`, so clearing it clears the Window
        # bit too and it has to be put back. Setting `Qt.Window` alone would
        # leave the dialog bit standing and change nothing.
        dialog.setWindowFlags((flags & ~Qt.WindowType.Dialog)
                              | Qt.WindowType.Window)
    except Exception:                       # pragma: no cover
        # Decoration must never be load-bearing (INVARIANTS 10): a dialog
        # that cannot be detached is still a dialog the user can use.
        pass
    return dialog


class _DetachEveryDialog:
    """Application-wide filter: every dialog is a window the user can drag.

    Asked 2026-08-21: "the settings for your data settings window should be
    movable without moving the main window. this should be tru of all
    settings windows or any popup window from spacr."

    ONE PLACE, NOT ONE CALL SITE PER DIALOG. :func:`detach_from_window_manager`
    already existed and was being called from six files while more than
    twenty others opened dialogs without it -- which is what "all settings
    windows" cannot be built out of. A rule applied by hand is a rule that
    holds until the next dialog is written.

    HOOKED ON `Polish`, WHICH IS THE ONLY MOMENT THAT WORKS.
    `setWindowFlags` on a widget that is already visible destroys and
    recreates its native window, so Qt hides it -- detaching on `Show` makes
    the dialog flash or vanish. `Polish` is delivered during the first show
    sequence and BEFORE the window is mapped, so the flags are already right
    when it appears. Measured order on PySide6: WinIdChange, Polish, Show,
    ShowToParent.

    Each dialog is detached ONCE. The flag change is idempotent, but doing
    it repeatedly on a dialog that is shown, hidden and shown again would
    recreate the native window every time and lose its position -- which is
    the thing this exists to protect.
    """

    def __init__(self):
        self._done: set = set()

    def eventFilter(self, obj, event):        # noqa: N802 - Qt naming
        try:
            from PySide6.QtCore import QEvent
            from PySide6.QtWidgets import QDialog

            if event.type() == QEvent.Type.Polish and isinstance(obj, QDialog):
                key = id(obj)
                if key not in self._done:
                    self._done.add(key)
                    detach_from_window_manager(obj)
        except Exception:                     # pragma: no cover
            # INVARIANTS 10 again: a dialog that cannot be detached is still
            # a dialog. This filter sees every event in the application and
            # must never be the reason one of them is lost.
            pass
        return False


#: Kept alive for the life of the application. An event filter that is
#: garbage collected stops filtering, silently.
_DETACHER = None


def detach_all_dialogs(app) -> bool:
    """Install the application-wide detacher. Returns True if it installed.

    Idempotent: calling it twice leaves one filter, because two would do the
    same work twice and the second would find every dialog already done.
    """
    global _DETACHER
    if app is None:
        return False
    if _DETACHER is not None:
        return False
    try:
        from PySide6.QtCore import QObject

        # ONE CLASS, NOT A MIX-IN, and that is a correctness fix rather than
        # a style one. `class F(QObject, _DetachEveryDialog)` puts QObject
        # first in the MRO, so QObject's own `eventFilter` -- which returns
        # False and does nothing -- wins over the one below it. The filter
        # installed, reported success, and silently never fired.
        class _Filter(QObject):
            def __init__(self):
                super().__init__()
                self._inner = _DetachEveryDialog()

            def eventFilter(self, obj, event):    # noqa: N802 - Qt naming
                return self._inner.eventFilter(obj, event)

        _DETACHER = _Filter()
        app.installEventFilter(_DETACHER)
        return True
    except Exception:                         # pragma: no cover
        _DETACHER = None
        return False
