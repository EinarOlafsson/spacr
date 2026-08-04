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
from typing import List, Optional, Sequence

from PySide6.QtCore import QEvent, QMimeData, QObject, Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent
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


def install_for(target: QWidget, app_key: str, screen: QWidget = None) -> bool:
    """Attach ``app_key``'s drop policy to ``target``. Never raises.

    The one line a screen adds to accept drops. Which policy that is comes
    from :func:`spacr.qt.dnd_handlers.get_handler`, so a screen never names a
    handler class and a screen with no declared policy still gets the
    source-folder fallback.

    Failure is a missing convenience, not a broken screen — a Qt build with no
    drag-and-drop, or a handler whose import fails, must not stop the screen
    being constructed. It is logged and the screen goes up without a dropzone.

    :param target: the widget that receives the drag/drop events.
    :param app_key: the registered app key, e.g. ``"graph_builder"``.
    :param screen: the object handed to ``handler.apply``; ``target`` when
        omitted.
    :returns: whether the dropzone was installed.
    """
    try:
        from .dnd_handlers import get_handler
        install_dropzone(target, get_handler(app_key), screen or target)
        return True
    except Exception:
        LOG.debug("no dropzone installed for %s", app_key, exc_info=True)
        return False


class _DropzoneFilter(QObject):
    """Event filter that routes drag/drop events on ``target`` into
    the :class:`DropHandler` attached to it."""

    def __init__(self, target: QWidget):
        # QObject parenting can synchronously deliver a ChildAdded event to
        # the target.  Set this first so eventFilter is fully initialized even
        # during super().__init__ (standalone tool screens exposed this race).
        self._target = target
        super().__init__(target)   # parent → auto-cleanup

    def eventFilter(self, obj, event):    # noqa: N802  (Qt naming)
        # `getattr`, not `self._target`, and the reason is not defensiveness
        # for its own sake. Qt goes on delivering events to a filter after the
        # target's C++ half is gone, and PySide6 clears the Python wrapper's
        # __dict__ when that happens -- so `self._target` raises AttributeError
        # from INSIDE the Qt event loop, which prints
        #
        #     Error calling Python override of QObject::eventFilter()
        #     AttributeError: '_DropzoneFilter' object has no attribute '_target'
        #
        # once per delivered event, and cannot be caught by any caller because
        # there is no Python caller. A filter whose target is gone has nothing
        # to filter, so declining the event is both correct and quiet.
        #
        # The same shape as `RunHandle.is_running` swallowing "Internal C++
        # object already deleted": the destroyed wrapper IS the answer, not an
        # error condition.
        target = getattr(self, "_target", None)
        if target is None or obj is not target:
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
        # Tell the drag source the drop landed as soon as we know we have
        # something to do with it. Doing this only at the very end meant a
        # settings-CSV-only drop (which IS handled below) was reported back
        # to the OS as rejected.
        event.acceptProposedAction()
        handler: DropHandler = self._target._dnd_handler
        screen = self._target._dnd_screen

        # Split: CSV → settings import (universal); anything else →
        # per-module handler.
        # A CSV is a universal settings import only on screens that actually
        # expose the settings importer.  Special-purpose screens (Plate Queue,
        # Batch Runner, Import Project) give CSVs their own meaning and must
        # receive them through their handler instead of losing them to a no-op.
        csvs = [p for p in paths
                if p.suffix.lower() == ".csv" and p.is_file()
                and hasattr(screen, "apply_settings_dict")]
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
                    _report_drop_problem(
                        screen, p, f"The drop handler failed: {e}",
                        "Check that the path is readable and that its contents "
                        "match this module, then try again.",
                    )
            else:
                alternatives = handler.suggest_alternatives(p)
                suggestion = (
                    "Choose one of the compatible nearby paths shown in the "
                    "dialog."
                    if alternatives else
                    "Open this module's source setting and choose a file or "
                    "folder matching the required layout."
                )
                _report_drop_problem(
                    screen, p, handler.error_message(p), suggestion,
                    alternatives=alternatives,
                )
                if alternatives:
                    pick = suggest_alternatives_dialog(
                        screen, p, alternatives,
                        why=handler.error_message(p),
                    )
                    if pick is not None:
                        try:
                            handler.apply(pick, screen)
                        except Exception as e:
                            _report_drop_problem(
                                screen, pick, f"The drop handler failed: {e}",
                                "Check that the path is readable and try again.",
                            )
                else:
                    QMessageBox.information(
                        screen, "Nothing to drop into",
                        f"{handler.error_message(p)}\n\nSuggestion: {suggestion}",
                    )


def _find_console(screen):
    """Return the nearest spaCR console, including the host app's console."""
    console = getattr(screen, "_console", None)
    if console is not None:
        return console
    try:
        window = screen.window()
    except Exception:
        return None
    console = getattr(window, "_console", None)
    if console is not None:
        return console
    # Standalone tool screens are hosted alongside AppScreens. Prefer the
    # most recently visited screen so rejected drops never disappear merely
    # because the tool itself has no embedded console.
    screens = getattr(window, "_screens", {}) or {}
    visit_order = list(getattr(window, "_visit_order", []) or [])
    for key in reversed(visit_order + list(screens)):
        candidate = screens.get(key)
        console = getattr(candidate, "_console", None)
        if console is not None:
            return console
    try:
        from spacr.qt.widgets.console_panel import ConsolePanel
        consoles = window.findChildren(ConsolePanel)
        if consoles:
            return consoles[-1]
    except Exception:
        pass
    return None


def _report_drop_problem(screen, path: Path, reason: str, suggestion: str,
                         alternatives: Sequence[Path] = ()) -> str:
    """Print an actionable rejected-drop report and optionally ask the AI."""
    lines = [
        f"[drop rejected] {path}",
        f"Reason: {reason}",
        f"Suggestion: {suggestion}",
    ]
    if alternatives:
        lines.append(
            "Compatible nearby paths: " +
            ", ".join(str(item) for item in alternatives)
        )
    message = "\n".join(lines) + "\n"
    LOG.warning(message.rstrip())
    console = _find_console(screen)
    displayed_inline = False
    if console is not None:
        append = getattr(console, "append_error", None) or getattr(
            console, "append_stdout", None)
        if append is not None:
            append(message)
            displayed_inline = True
        try:
            from spacr.qt.ai.settings import get_route_errors_through_ai
            provider = console._current_provider()
            ai_active = getattr(console, "_ai_active", False)
            if callable(ai_active):
                ai_active = ai_active()
            if (get_route_errors_through_ai()
                    and bool(ai_active)
                    and provider is not None):
                console.open_error_flow(
                    message,
                    active_app=getattr(screen, "app_key", None),
                    show_raw=False,
                )
        except Exception:
            LOG.debug("Could not route rejected drop through AI",
                      exc_info=True)
    # Standalone tools use a read-only summary/log pane instead of an
    # AppScreen ConsolePanel. Put the same actionable text there as well.
    if not displayed_inline:
        for attr in ("_summary", "_log", "_console_text"):
            widget = getattr(screen, attr, None)
            append = getattr(widget, "appendPlainText", None)
            if callable(append):
                append(message.rstrip())
                displayed_inline = True
                break
    status = getattr(screen, "_set_status", None)
    if callable(status):
        try:
            status(f"Drop rejected: {reason} Suggestion: {suggestion}")
        except Exception:
            pass
    return message


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
        # spaCR's own save_settings writes Key/Value columns; other tools
        # (and older spaCR CSVs) use setting_key/setting_value. load_settings
        # RAISES on a column mismatch rather than returning something
        # non-dict, so the second form has to be tried in its own except —
        # otherwise the fallback was unreachable and every
        # setting_key/setting_value CSV was reported as a failed import.
        try:
            loaded = load_settings(str(path),
                                     setting_key="Key",
                                     setting_value="Value")
        except Exception:
            loaded = None
        if not isinstance(loaded, dict):
            loaded = load_settings(str(path))
        if isinstance(loaded, dict):
            n = screen.apply_settings_dict(loaded)
            if hasattr(screen, "_console"):
                screen._console.append_stdout(
                    f"[drop] imported {n} settings from {path.name}\n"
                )
    except Exception as e:
        _report_drop_problem(
            screen, path, f"Settings CSV import failed: {e}",
            "Export a settings CSV from spaCR, or verify that the file has "
            "Key/Value or setting_key/setting_value columns.",
        )
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
