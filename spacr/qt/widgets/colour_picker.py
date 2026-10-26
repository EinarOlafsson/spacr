"""The one colour picker the GUI uses, with the platform dialog turned off.

WHY THIS MODULE EXISTS AT ALL
-----------------------------
``QColorDialog.getColor`` defaults to the *platform's* colour chooser. On a
GNOME desktop with ``xdg-desktop-portal`` running, that is not a Qt widget at
all: Qt asks the portal, the portal starts (or wakes) the GTK implementation
over D-Bus, and the dialog can be slow to appear. Every colour picker in the
tree was reached through an unguarded ``getColor`` — six of them, none passing
``DontUseNativeDialog``.

Qt's own dialog opens immediately, looks the same on every platform, and
follows the application palette — which the GTK one does not.

The portal round trip **cannot be reproduced
headless**: an offscreen Qt never asks the portal, so no test in this suite can
observe the stall or its absence. The restyle work itself is free
(``set_line_style`` 0.000 s on a 1,200
point plot) so the wait is in the dialog — and "the portal is what the wait is"
is a named, checkable hypothesis to be confirmed on a real display, not
something this file proves. What IS proven here, by
``tests/qt/test_colour_picker.py``, is that no call site can ask for the
platform dialog any more.

Use it instead of ``QColorDialog.getColor``, so no call site has to remember
the option. A grep test enforces this.
"""
from __future__ import annotations

from typing import Optional, Union

from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QWidget


def pick_colour(parent: Optional[QWidget] = None,
                initial: Union[QColor, str, None] = None,
                title: Optional[str] = None) -> QColor:
    """Ask the user for a colour, using Qt's own dialog.

    :param parent: dialog parent, as for :meth:`QColorDialog.getColor`.
    :param initial: the colour to open on — a :class:`QColor`, a string
        Qt understands (``"#ff0000"``, ``"red"``), or None for white.
    :param title: window title; Qt's default when omitted.
    :returns: the chosen :class:`QColor`, or an **invalid** ``QColor`` when
        the user cancelled.

    Returning an invalid colour rather than ``None`` is deliberate: every
    existing call site already tests ``colour.isValid()`` before using it, so
    adopting this helper is a one-line change at each and cannot silently
    turn a cancel into a colour.
    """
    if isinstance(initial, QColor):
        start = QColor(initial)
    elif initial is None:
        start = QColor("#ffffff")
    else:
        start = QColor(str(initial))
    if not start.isValid():
        start = QColor("#ffffff")
    from ..dialogs import detach_from_window_manager

    dialog = QColorDialog(start, parent)
    dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
    if title:
        dialog.setWindowTitle(title)
    detach_from_window_manager(dialog)
    dialog.ensurePolished()
    dialog.adjustSize()
    if parent is not None:
        opening = dialog.frameGeometry()
        opening.moveCenter(parent.window().frameGeometry().center())
        dialog.move(opening.topLeft())
    try:
        if dialog.exec() == QColorDialog.DialogCode.Accepted:
            return QColor(dialog.selectedColor())
        return QColor()
    finally:
        dialog.deleteLater()
