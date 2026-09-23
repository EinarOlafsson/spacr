"""Shared appearance and visible-path checks for the current tutorial pass."""
from __future__ import annotations

import re
from pathlib import Path

CAPTURE_THEME = "dark"
CAPTURE_BACKDROP = "blobs"
_PRIVATE_PATH = re.compile(r"(?:/home/[^/\s]+|/Users/[^/\s]+|[A-Za-z]:[\\/]Users[\\/][^\\/\s]+|/mnt/[^\s<>\"']+|/nas_mnt(?:/[^\s<>\"']*)?)")


def configure_appearance(theme=CAPTURE_THEME, backdrop=CAPTURE_BACKDROP):
    """Set the requested recording appearance in the isolated Qt store."""
    if (theme, backdrop) != (CAPTURE_THEME, CAPTURE_BACKDROP):
        raise ValueError("Tutorial captures require dark mode and the Blobs backdrop")
    from spacr.qt import preferences as prefs

    prefs.set_theme(theme)
    prefs.set_ambient_animation(backdrop)


def verify_appearance(window):
    """Check the effective palette and the backdrop widgets being painted."""
    from PySide6.QtGui import QPalette

    from spacr.qt import preferences as prefs
    from spacr.qt.widgets.ambient import AmbientWidget
    from spacr.qt.widgets.dna_rain import DnaRainWidget

    if prefs.resolve_effective_theme() != CAPTURE_THEME:
        raise RuntimeError("Capture refused: the effective theme is not dark")
    if not prefs.get_ambient_enabled() or prefs.get_ambient_animation() != CAPTURE_BACKDROP:
        raise RuntimeError("Capture refused: Blobs is disabled or another backdrop is selected")
    if window.palette().color(QPalette.Window).lightness() >= 128:
        raise RuntimeError("Capture refused: the window is painting a light palette")
    visible = [widget for widget in window.findChildren(AmbientWidget) if widget.isVisible()]
    if not visible or any(widget.theme() != CAPTURE_BACKDROP for widget in visible):
        raise RuntimeError("Capture refused: visible backdrop widgets are not Blobs")
    if not any(widget.frames_painted > 0 for widget in visible):
        raise RuntimeError("Capture refused: the Blobs backdrop has not painted a frame")
    if any(widget.isVisible() for widget in window.findChildren(DnaRainWidget)):
        raise RuntimeError("Capture refused: DNA rain covers the requested Blobs backdrop")
    return {"theme": CAPTURE_THEME, "backdrop": CAPTURE_BACKDROP,
            "painted_frames": sum(widget.frames_painted for widget in visible)}


def exclude_special_backdrops(window):
    """Expose the shared Blobs backdrop in the recording namespace.

    Map Barcodes normally draws its own DNA rain over the window backdrop.
    Hiding that decorative layer is a capture presentation choice, like
    excluding News. No application code, controls or analysis results change.
    """
    from spacr.qt.widgets.dna_rain import DnaRainWidget

    hidden = []
    for widget in window.findChildren(DnaRainWidget):
        if widget.isVisible():
            widget.hide()
            hidden.append(type(widget).__name__)
    return hidden


def exclude_release_history(window):
    """Omit the historical News aside from current-version recordings.

    This is a capture presentation choice, like omitting recent-run history.
    Application sources, release notes and workflow results are unchanged.
    """
    from spacr.qt.widgets.home import NewsPanel

    panels = window.findChildren(NewsPanel)
    for panel in panels:
        panel.hide()
    return len(panels)


def verify_visible_paths(windows, prepared_root):
    """Refuse visible personal/mounted paths before saving a tutorial frame.

    This inspects Qt text surfaces, including visible table cells. Images and
    external desktop windows still require the sampled-frame visual review.
    """
    from PySide6.QtCore import Qt
    from spacr import __version__
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QComboBox,
        QLabel,
        QLineEdit,
        QListView,
        QPlainTextEdit,
        QTextEdit,
        QWidget,
    )

    allowed = str(Path(prepared_root).absolute())
    if _PRIVATE_PATH.search(allowed):
        raise ValueError("Use a neutral prepared capture directory, for example /tmp/spacr-tutorials")
    for window in windows:
        for widget in [window, *window.findChildren(QWidget)]:
            if not widget.isVisible():
                continue
            texts = [widget.windowTitle()]
            if isinstance(widget, (QLabel, QLineEdit)):
                texts.append(widget.text())
            elif isinstance(widget, (QTextEdit, QPlainTextEdit)):
                texts.append(widget.toPlainText())
            elif isinstance(widget, QComboBox):
                texts.append(widget.currentText())
            if isinstance(widget, QAbstractItemView) and widget.model() is not None:
                model = widget.model()
                index = widget.indexAt(widget.viewport().rect().topLeft())
                first_row = max(0, index.row())
                for row in range(first_row, model.rowCount(widget.rootIndex())):
                    row_visible = False
                    # QListWidget's internal model exposes columnCount as a
                    # private C++ method in PySide; a list always has one.
                    columns = 1 if isinstance(widget, QListView) else model.columnCount(widget.rootIndex())
                    for column in range(columns):
                        cell = model.index(row, column, widget.rootIndex())
                        if widget.visualRect(cell).intersects(widget.viewport().rect()):
                            row_visible = True
                            texts.append(str(model.data(cell, Qt.DisplayRole) or ""))
                    if not row_visible and row > first_row:
                        break
            for text in texts:
                plain_text = re.sub(r"<[^>]+>", "", text)
                releases = re.findall(
                    r"\bspaCR[\s-]+(\d+\.\d+\.\d+(?:\.\d+)?)", plain_text, re.I)
                if any(version != __version__ for version in releases):
                    raise RuntimeError(
                        "Capture refused: visible text references another spaCR release")
                for match in _PRIVATE_PATH.finditer(text):
                    path = match.group(0)
                    if path != allowed and not path.startswith(allowed + "/"):
                        raise RuntimeError(
                            "Capture refused: a visible text surface contains a personal "
                            "or mounted path outside the prepared capture directory")
