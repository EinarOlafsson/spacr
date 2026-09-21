"""A Refresh button for every live preview: read src again and reload.

Item 464. The previews load their first field from ``src`` once, on their
own. When the folder changes under the same path, or the automatic load
picked nothing because the path was wrong at the time, the user needs a way
to say "look again". This is that button, one implementation for all of
them, because each preview only differs in the name of its loader.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QToolButton

from ..i18n import tr

#: The loader each preview panel exposes, in the order they are tried: the
#: Cellpose live preview, the crop preview, the track preview and the
#: motility preview.
LOADERS = ("load_source_async", "load_array_async", "load_sequence_async",
           "load_folder_async")


def _say(screen: Any, text: str) -> None:
    """Write one line to the screen's console, if it has one.

    :param screen: the module screen.
    :param text: the line.
    """
    console = getattr(screen, "_console", None)
    append = getattr(console, "append_stdout", None)
    if callable(append):
        append(text + "\n")


def current_src(screen: Any) -> str:
    """The source path the screen's form holds right now.

    :param screen: the module screen.
    :returns: the path, or ``""``.
    """
    reader = getattr(screen, "_settings_src_path", None)
    if callable(reader):
        try:
            return str(reader() or "").strip()
        except Exception:
            return ""
    return ""


def reload_from_src(screen: Any, panel: Any) -> bool:
    """Check the screen's src and reload ``panel`` from it.

    :param screen: the module screen whose form holds ``src``.
    :param panel: the preview panel to reload.
    :returns: whether a load was started.
    """
    src = current_src(screen)
    if not src or src in {"path", "/path", "/path/to/src"}:
        _say(screen, tr("Refresh: the source setting is empty."))
        return False
    if not os.path.exists(src):
        _say(screen, tr("Refresh: {path} does not exist.").format(path=src))
        return False
    for name in LOADERS:
        loader = getattr(panel, name, None)
        if callable(loader):
            if hasattr(panel, "_auto_loaded_src"):
                panel._auto_loaded_src = src
            _say(screen, tr("Refresh: reloading the preview from {path}").format(
                path=src))
            return bool(loader(src))
    return False


def install_refresh_button(screen: Any, card: Any, panel: Any
                           ) -> Optional[QToolButton]:
    """Add a Refresh button to a preview card's title row.

    :param screen: the module screen.
    :param card: the preview's :class:`~spacr.qt.widgets.card.Card`.
    :param panel: the preview panel it reloads.
    :returns: the button, or ``None`` when the card has no title row to take it.
    """
    add = getattr(card, "add_title_action", None)
    if not callable(add) or getattr(card, "_refresh_button", None) is not None:
        return getattr(card, "_refresh_button", None)
    button = QToolButton(card)
    button.setObjectName("PreviewRefreshButton")
    button.setText(tr("Refresh"))
    button.setCursor(Qt.PointingHandCursor)
    button.setToolTip(tr(
        "Read the source setting again and reload this preview from it. Use "
        "it after changing the folder's contents or fixing the path."))
    button.clicked.connect(lambda _checked=False: reload_from_src(screen, panel))
    add(button)
    card._refresh_button = button
    return button
