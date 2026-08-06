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
