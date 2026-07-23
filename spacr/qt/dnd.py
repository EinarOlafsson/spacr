"""
Drag-and-drop system for AppScreens.

Design:

* :class:`DropHandler` — per-module policy: what folders/files this
  screen accepts, how to fix a "close-but-not-quite" drop, and what
  to do once a drop is accepted.
* :func:`install_dropzone` — attaches Qt drop event handlers to any
  widget (usually the AppScreen itself) and wires them to a
  :class:`DropHandler`.
* :func:`suggest_alternatives_dialog` — the "did you mean X?"
  chooser shown when the dropped folder can't be used as-is but a
  sibling / child folder can.

Behaviour common to every module:

* Dropping a ``*.csv`` file → treat as a settings CSV and call the
  screen's ``apply_settings_dict`` (imports settings, doesn't
  overwrite the source folder).
* Dropping a folder → hand off to the module's ``DropHandler``.
  If it's a good fit, the handler calls ``screen._set_src`` (or
  equivalent). If it's a near-miss the user gets the "did you mean"
  dialog.
* Dropping multiple folders → the handler is called once per folder
  in the order the OS delivers them. Modules that don't handle
  multi-drop degrade to first-only.

Per-module policies live in :mod:`spacr.qt.dnd_handlers`.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from PySide6.QtCore import QEvent, QMimeData, QObject, QUrl, Qt
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QLabel, QListWidget, QListWidgetItem,
    QMessageBox, QVBoxLayout, QWidget,
)

LOG = logging.getLogger("spacr.qt.dnd")

# File extensions that count as images for "does this folder have
# images?" checks. Keep in sync with spacr.io's readers.
IMAGE_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".czi",
              ".nd2", ".lif")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DropHandler(ABC):
    """Per-module drop policy.

    Subclasses implement:
        can_accept(path)          — is this path good to go?
        apply(path, screen)       — wire it into the screen.
    And optionally override:
        suggest_alternatives(p)   — return nearby folders that DO fit.
        error_message(p)          — return the "why not?" string.
        accepts_multiple()        — True if multi-folder drops make sense.
    """

    # -- public API subclasses implement -----------------------------------
    @abstractmethod
    def can_accept(self, path: Path) -> bool:
        """Return True if ``path`` (folder OR file) is usable as-is."""

    @abstractmethod
    def apply(self, path: Path, screen) -> None:
        """Wire ``path`` into ``screen`` (set src, populate settings, etc.)."""

    def suggest_alternatives(self, path: Path) -> List[Path]:
        """When ``can_accept`` returns False, return sibling/child folders
        that WOULD be accepted so the UI can prompt "did you mean…".

        Default: no suggestions.
        """
        return []

    def error_message(self, path: Path) -> str:
        """Human-friendly explanation for why ``path`` can't be used."""
        return f"This module can't use {path.name!r}."

    def accepts_multiple(self) -> bool:
        """Return True to be called per-folder on multi-item drops."""
        return False


def install_dropzone(target: QWidget, handler: DropHandler,
                       screen: QWidget) -> None:
    """Wire ``target`` to accept drops routed through ``handler``.

    Typically called from ``AppScreen.__init__``: ``target`` is
    ``self`` and ``screen`` is also ``self``. Splitting them lets
    non-AppScreen widgets install a dropzone that acts on a
    different owner (e.g. a specific input row).

    :param target: the QWidget that receives drag/drop events.
    :param handler: the module's DropHandler policy.
    :param screen: the widget passed to ``handler.apply`` — usually
        the AppScreen.
    """
    target.setAcceptDrops(True)

    # Store the handler + owning-screen on the widget itself so the
    # event filter can look them up without capturing them in a
    # closure that would keep the target alive after destruction.
    target._dnd_handler = handler
    target._dnd_screen = screen
    # Filter is parented to target — Qt cleans it up when target dies.
    f = _DropzoneFilter(target)
    target.installEventFilter(f)


class _DropzoneFilter(QObject):
    """Event filter that routes drag/drop events on ``target`` into
    the :class:`DropHandler` attached to it."""

    def __init__(self, target: QWidget):
        super().__init__(target)   # parent → auto-cleanup
        self._target = target

    def eventFilter(self, obj, event):    # noqa: N802  (Qt naming)
        if obj is not self._target:
            return False
        et = event.type()
        if et == QEvent.DragEnter:
            self._on_drag_enter(event)
            return True
        if et == QEvent.DragMove:
            event.acceptProposedAction()
            return True
        if et == QEvent.Drop:
            self._on_drop(event)
            return True
        return False

    # -- handlers ----------------------------------------------------------
    def _on_drag_enter(self, event: QDragEnterEvent) -> None:
        mime = event.mimeData()
        if _mime_has_local_paths(mime):
            event.acceptProposedAction()

    def _on_drop(self, event: QDropEvent) -> None:
        paths = _mime_local_paths(event.mimeData())
        if not paths:
            return
        handler: DropHandler = self._target._dnd_handler
        screen = self._target._dnd_screen

        # Split: CSV → settings import (universal); anything else →
        # per-module handler.
        csvs = [p for p in paths
                if p.suffix.lower() == ".csv" and p.is_file()]
        others = [p for p in paths if p not in csvs]

        for p in csvs:
            _apply_settings_csv(p, screen)

        if not others:
            return

        if not handler.accepts_multiple():
            others = others[:1]

        for p in others:
            if handler.can_accept(p):
                try:
                    handler.apply(p, screen)
                except Exception as e:
                    QMessageBox.warning(screen, "Drop failed", str(e))
            else:
                alternatives = handler.suggest_alternatives(p)
                if alternatives:
                    pick = suggest_alternatives_dialog(
                        screen, p, alternatives,
                        why=handler.error_message(p),
                    )
                    if pick is not None:
                        try:
                            handler.apply(pick, screen)
                        except Exception as e:
                            QMessageBox.warning(screen, "Drop failed",
                                                 str(e))
                else:
                    QMessageBox.information(
                        screen, "Nothing to drop into",
                        handler.error_message(p),
                    )
        event.acceptProposedAction()


# ---------------------------------------------------------------------------
# Mime helpers
# ---------------------------------------------------------------------------

def _mime_has_local_paths(mime: QMimeData) -> bool:
    if not mime.hasUrls():
        return False
    return any(u.isLocalFile() for u in mime.urls())


def _mime_local_paths(mime: QMimeData) -> List[Path]:
    return [Path(u.toLocalFile()) for u in mime.urls()
            if u.isLocalFile()]


# ---------------------------------------------------------------------------
# Universal CSV → settings importer
# ---------------------------------------------------------------------------

def _apply_settings_csv(path: Path, screen) -> None:
    """Load a settings CSV and push into ``screen.apply_settings_dict``.

    Silent no-op if the screen doesn't have that method (AnnotateScreen,
    MakeMasksScreen — they don't use the SettingsWidgets model).
    """
    if not hasattr(screen, "apply_settings_dict"):
        return
    try:
        from spacr.utils import load_settings
        loaded = load_settings(str(path),
                                 setting_key="Key",
                                 setting_value="Value")
        if not isinstance(loaded, dict):
            loaded = load_settings(str(path))
        if isinstance(loaded, dict):
            n = screen.apply_settings_dict(loaded)
            if hasattr(screen, "_console"):
                screen._console.append_stdout(
                    f"[drop] imported {n} settings from {path.name}\n"
                )
    except Exception as e:
        QMessageBox.warning(screen, "CSV import failed", str(e))


# ---------------------------------------------------------------------------
# "Did you mean X?" dialog
# ---------------------------------------------------------------------------

def suggest_alternatives_dialog(
    parent, original: Path, alternatives: Sequence[Path], why: str = "",
) -> Optional[Path]:
    """Modal that lets the user pick from ``alternatives``.

    :returns: the chosen Path, or None if cancelled.
    """
    dlg = QDialog(parent)
    dlg.setWindowTitle("Did you mean…")
    dlg.setMinimumWidth(520)
    layout = QVBoxLayout(dlg)

    header = QLabel(
        f"<b>{original.name}</b> can't be used as-is."
        + (f"<br><span style='color:gray;'>{why}</span>" if why else "")
        + "<br><br>Nearby folders that WOULD work:"
    )
    header.setTextFormat(Qt.RichText)
    header.setWordWrap(True)
    layout.addWidget(header)

    lst = QListWidget()
    for alt in alternatives:
        item = QListWidgetItem(str(alt))
        lst.addItem(item)
    lst.setCurrentRow(0)
    layout.addWidget(lst, 1)

    buttons = QDialogButtonBox(
        QDialogButtonBox.Ok | QDialogButtonBox.Cancel
    )
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    if dlg.exec() != QDialog.Accepted:
        return None
    row = lst.currentRow()
    if row < 0:
        return None
    return alternatives[row]


# ---------------------------------------------------------------------------
# Filesystem helpers reused by handlers
# ---------------------------------------------------------------------------

def has_images_in(path: Path, min_count: int = 1,
                    exts: Sequence[str] = IMAGE_EXTS) -> bool:
    """Return True if ``path`` contains at least ``min_count`` image
    files at its top level (does not recurse)."""
    if not path.is_dir():
        return False
    count = 0
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in exts:
            count += 1
            if count >= min_count:
                return True
    return False


def find_image_folders_nearby(path: Path, max_depth: int = 1,
                                min_count: int = 1) -> List[Path]:
    """Search parent + immediate children of ``path`` for folders that
    contain images. Excludes ``path`` itself if it already qualifies.

    Handy for the "did you mean X?" prompt when the user drops the
    wrong sibling of a plate folder.
    """
    hits: List[Path] = []
    # One level up: check siblings
    if path.parent and path.parent.is_dir():
        for sib in path.parent.iterdir():
            if sib.is_dir() and sib != path and has_images_in(sib, min_count):
                hits.append(sib)
    # One level down: check immediate children
    if path.is_dir():
        for child in path.iterdir():
            if child.is_dir() and has_images_in(child, min_count):
                hits.append(child)
    return hits


def sample_image_names(path: Path, n: int = 8,
                         exts: Sequence[str] = IMAGE_EXTS) -> List[Path]:
    """Return up to ``n`` image paths from ``path`` — used by the
    filename-regex preview in the mask handler."""
    if not path.is_dir():
        return []
    out: List[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in exts:
            out.append(child)
            if len(out) >= n:
                break
    return out
