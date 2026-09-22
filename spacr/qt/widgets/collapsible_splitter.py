"""Drag an edge to resize a pane; collapsing is where the drag stops.

Item 471, slice B. Two requests that turn out to be one: every container
collapses ("whenever anything is collapsed it should auto lock to the bottom
of the container it is in"), and every container can be "expanded or
shrinkable by draging the edges". A :class:`~PySide6.QtWidgets.QSplitter`
already does both -- a handle to drag, and a child that can be taken down to
nothing -- so both halves live here as ONE mechanism, and anything that
collapses in a splitter also resizes, and the other way round.

THE THREE PIECES, AND WHAT SLICE C REUSES
-----------------------------------------

:class:`CollapsibleSplitter`
    A ``QSplitter`` whose children are registered as named panes::

        split = CollapsibleSplitter(Qt.Vertical, persist_key="mask::shell")
        split.add_pane(card, "Figures", folder=card.folder, extent=420)
        split.add_pane(console_wrap, "Console", folder=console_folder)
        split.add_pane(settings, "Settings", mode=EDGE, fold_key="mask/Settings")

    A pane is one of three kinds:

    * ``HEADER`` -- it has a :class:`~spacr.qt.widgets.foldable.Folder` (a
      clickable heading over a body). Collapsed, its body is hidden, the pane
      shrinks to its heading, and the heading sits at the BOTTOM of the room
      the pane is given (:func:`lock_folded_to_bottom`). This is the kind for
      anything with a name to click: the console, System, a figure panel.
    * ``EDGE`` -- it collapses to nothing and the handle beside it becomes the
      strip that brings it back: click the handle to hide or show the pane,
      drag it to resize. The kind for a column with no heading of its own (the
      settings column, which collapses to the left).
    * plain (``mode=None``, no folder) -- resizable, never collapsed.

    Dragging an expanded pane below its minimum collapses it (Qt's own
    collapse, turned into the pane's folded state). Collapsed HEADER panes are
    held at their heading's height, so a window resize never hands them room
    they would show as a gap. When nothing in the splitter is left expanded,
    the FIRST collapsed pane takes the spare room and, locked to the bottom,
    stacks every heading at the bottom of the container.

    Sizes are remembered per pane NAME (not per index, so adding a pane to a
    screen does not scramble the old layout) under ``persist_key``, in the
    preferences store. Only what the user dragged is stored; a collapse made
    on the user's behalf is never stored as a size.

    API: ``add_pane(widget, name, *, folder=None, mode=None, stretch=1,
    extent=0, minimum=None, focus=False, fold_key="", index=None) -> Pane``;
    ``pane(name)``, ``panes()``, ``is_collapsed(name)``,
    ``set_collapsed(name, collapsed, *, by_user=False)``,
    ``toggle_pane(name, *, by_user=True)``, ``rebalance(grow=None)``, and the
    ``pane_toggled(name, collapsed, by_user)`` signal. A widget that is later
    re-wrapped in a container which is put in its place (the settings search
    strip does this to the settings column) keeps its registration: the
    container inherits it.

:class:`FocusCollapse`
    The automatic half. Some widgets are FOCUS widgets (a live preview, the
    figures panel): while any of them is on screen, the TARGET panes collapse
    (console, System, the buttons, the settings column), and when the last
    one goes they come back. It never fights the user: a target the user
    expands by hand is PINNED open and left alone until the VIEW ends. See
    the class for what a view is.

:func:`lock_folded_to_bottom`
    For a folded panel that is NOT in a splitter (a section in a scroll area,
    a figure tile in a column): keeps its heading at the bottom of whatever
    room its layout gives it, which is the "auto lock to the bottom" rule on
    its own.
"""
from __future__ import annotations

import json
import logging
from functools import partial
from typing import Callable, List, Optional

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPolygon
from PySide6.QtWidgets import (QSizePolicy, QSpacerItem, QSplitter,
                               QSplitterHandle, QWidget)

from ..i18n import tr

LOG = logging.getLogger("spacr.qt.collapsible_splitter")

#: A pane with a clickable heading: collapsed, it is that heading, at the
#: bottom of its room.
HEADER = "header"

#: A pane that collapses to nothing; the handle beside it is its strip.
EDGE = "edge"

#: Qt's "no maximum". Named so the reset reads as a reset.
UNLIMITED = 16777215

#: Width of a handle between panes, in pixels. Wide enough to grab; only a
#: one-pixel line in its middle is drawn, so the gap looks as it did.
GRIP_PX = 6

#: Width of a handle that is also a collapsed pane's strip. Wider, because it
#: carries the arrow that says which way a click will send the pane.
EDGE_GRIP_PX = 12

#: A press that moves less than this is a click on the handle, not a drag.
CLICK_SLOP_PX = 3

#: Preferences key prefix for the remembered pane sizes.
_EXTENTS_PREFIX = "layout/panes"


def _alive(widget) -> bool:
    """Whether ``widget`` still has its C++ half."""
    if widget is None:
        return False
    try:
        from shiboken6 import isValid

        return bool(isValid(widget))
    except Exception:                                        # noqa: BLE001
        return True


def _settings():
    """The preferences store; see :func:`spacr.qt.preferences._settings`."""
    from ..preferences import _settings as _prefs_settings

    return _prefs_settings()


def get_pane_extents(persist_key: str) -> dict:
    """The sizes the user dragged the panes of ``persist_key`` to.

    :param persist_key: the splitter's key, e.g. ``"mask::runtime"``.
    :returns: ``{pane name: pixels}``; empty when nothing was ever dragged or
        the stored value is unreadable.
    """
    key = str(persist_key or "").strip()
    if not key:
        return {}
    try:
        raw = _settings().value(f"{_EXTENTS_PREFIX}/{key}", "")
        value = json.loads(raw) if raw else {}
    except Exception:                                        # noqa: BLE001
        return {}
    if not isinstance(value, dict):
        return {}
    out = {}
    for name, size in value.items():
        try:
            if int(size) > 0:
                out[str(name)] = int(size)
        except (TypeError, ValueError):
            continue
    return out


def set_pane_extents(persist_key: str, extents: dict) -> None:
    """Remember ``{pane name: pixels}`` for ``persist_key``.

    :param persist_key: the splitter's key.
    :param extents: ``{pane name: pixels}``; zero and negative sizes are
        dropped.
    """
    key = str(persist_key or "").strip()
    if not key:
        return
    try:
        clean = {str(k): int(v) for k, v in dict(extents or {}).items()
                 if int(v) > 0}
        _settings().setValue(f"{_EXTENTS_PREFIX}/{key}", json.dumps(clean))
    except Exception:                                        # noqa: BLE001
        LOG.debug("could not store the pane sizes", exc_info=True)


def lock_folded_to_bottom(container: QWidget, folder,
                          orientation=Qt.Vertical) -> None:
    """Keep ``container``'s heading at the bottom of its room while folded.

    A folded panel given more room than its heading needs -- the last pane
    in a splitter, a section stretched by its column -- used to lay its
    heading out in the middle of that room, which is the bug item 471 names
    ("when i collapse the console it collapses to the middle"). A stretch
    put ABOVE the heading while it is folded takes the spare room instead.

    :param container: the widget whose layout holds the heading (first) and
        the body.
    :param folder: its :class:`~spacr.qt.widgets.foldable.Folder`.
    :param orientation: ``Qt.Horizontal`` locks it to the left instead, for a
        panel that folds sideways.
    """
    state = {"spacer": None}

    def apply(shut: bool, _by_user: bool = True) -> None:
        """Add or take away the stretch in front of the heading."""
        layout = container.layout() if _alive(container) else None
        if layout is None:
            return
        spacer = state["spacer"]
        if shut and spacer is None:
            if orientation == Qt.Vertical:
                spacer = QSpacerItem(0, 0, QSizePolicy.Minimum,
                                     QSizePolicy.Expanding)
                layout.insertItem(0, spacer)
            else:
                spacer = QSpacerItem(0, 0, QSizePolicy.Expanding,
                                     QSizePolicy.Minimum)
                layout.addItem(spacer)
            state["spacer"] = spacer
        elif not shut and spacer is not None:
            layout.removeItem(spacer)
            state["spacer"] = None
        layout.invalidate()

    folder.add_listener(apply)
    if folder.shut:
        apply(True)


class Pane:
    """One registered child of a :class:`CollapsibleSplitter`.

    Made by :meth:`CollapsibleSplitter.add_pane`; each argument is kept as
    the attribute of the same name.

    :param widget: the splitter's direct child.
    :param name: stable English name; the key its size is stored under.
    :param mode: :data:`HEADER`, :data:`EDGE` or ``None``.
    :param folder: the heading's Folder, for a HEADER pane.
    :param stretch: 0 keeps the pane at its own size when the splitter grows.
    :param extent: the size it opens at; updated by every drag.
    :param minimum: its minimum while open, dropped while it is collapsed so
        it can shrink to its heading.
    :param focus: showing it also opens it (a live preview the user folded
        opens again when it is switched back on).
    :param fold_key: stores an EDGE pane's collapse across restarts.
    """

    def __init__(self, widget, name, mode, folder, stretch, extent, minimum,
                 focus, fold_key):
        """Hold the pane's description; see the class for each field."""
        self.widget = widget
        self.name = str(name)
        self.mode = mode
        self.folder = folder
        self.stretch = int(stretch)
        self.extent = int(extent or 0)
        self.minimum = minimum
        self.focus = bool(focus)
        self.fold_key = str(fold_key or "")
        self.edge_collapsed = False

    def is_collapsed(self) -> bool:
        """Whether the pane is collapsed right now."""
        if self.mode == HEADER and self.folder is not None:
            return bool(self.folder.shut)
        if self.mode == EDGE:
            return self.edge_collapsed
        return False


class _PaneHandle(QSplitterHandle):
    """A splitter handle that draws a thin line, and is a strip when needed.

    Painted here rather than by the stylesheet, because the theme's hover rule
    colours the WHOLE handle, and a grab area wide enough to find would turn
    into a blue bar. Beside an EDGE pane it also carries an arrow, and a
    click (a press that does not move) collapses or restores that pane.
    """

    def __init__(self, orientation, splitter):
        """Build the handle; ``splitter`` is its CollapsibleSplitter."""
        super().__init__(orientation, splitter)
        self.setAttribute(Qt.WA_Hover, True)
        self._pressed_at = None
        self._dragged = False

    def edge_pane(self) -> Optional[Pane]:
        """The EDGE pane this handle is the strip of, if any."""
        splitter = self.splitter()
        finder = getattr(splitter, "_edge_pane_beside", None)
        return finder(self) if callable(finder) else None

    def retranslate_dynamic_content(self, language=None) -> None:
        """Rewrite the tooltip in ``language``."""
        pane = self.edge_pane()
        if pane is None:
            self.setToolTip("")
            return
        name = tr(pane.name, language)
        if pane.is_collapsed():
            self.setToolTip(tr("Click to show {name} again, or drag to "
                               "resize it.", language, name=name))
        else:
            self.setToolTip(tr("Click to hide {name}, or drag to resize it.",
                               language, name=name))

    def enterEvent(self, event) -> None:                     # noqa: N802
        """Light the line up under the pointer."""
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:                     # noqa: N802
        """Put the line back."""
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:                # noqa: N802
        """Remember where a press began, to tell a click from a drag."""
        self._pressed_at = event.position().toPoint()
        self._dragged = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:                 # noqa: N802
        """A press that travels is a drag."""
        if self._pressed_at is not None:
            moved = event.position().toPoint() - self._pressed_at
            if moved.manhattanLength() > CLICK_SLOP_PX:
                self._dragged = True
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:              # noqa: N802
        """A click beside an EDGE pane collapses or restores it."""
        super().mouseReleaseEvent(event)
        clicked = (self._pressed_at is not None and not self._dragged
                   and event.button() == Qt.LeftButton)
        self._pressed_at = None
        pane = self.edge_pane() if clicked else None
        if pane is not None:
            self.splitter().toggle_pane(pane.name, by_user=True)

    def paintEvent(self, _event) -> None:                    # noqa: N802
        """Draw the one-pixel line and, beside an EDGE pane, its arrow."""
        try:
            from ..theme import active_palette

            palette = active_palette()
        except Exception:                                    # noqa: BLE001
            palette = {}
        hovered = self.underMouse()
        line = QColor(palette.get("accent" if hovered else "border_soft",
                                  "#4c8dff" if hovered else "#3a3f4b"))
        painter = QPainter(self)
        rect = self.rect()
        if self.orientation() == Qt.Horizontal:
            x = rect.center().x()
            painter.fillRect(x, rect.top(), 1, rect.height(), line)
        else:
            y = rect.center().y()
            painter.fillRect(rect.left(), y, rect.width(), 1, line)
        pane = self.edge_pane()
        if pane is not None:
            self._paint_arrow(painter, pane, palette, hovered)
        painter.end()

    def _paint_arrow(self, painter, pane, palette, hovered) -> None:
        """A small tab with a triangle pointing where a click sends the pane."""
        splitter = self.splitter()
        before = splitter.indexOf(pane.widget) < splitter.indexOf(
            splitter.widget(self._index()))
        collapsed = pane.is_collapsed()
        towards_start = before != collapsed
        rect = self.rect()
        painter.setRenderHint(QPainter.Antialiasing, True)
        tab = QColor(palette.get("surface", "#2a2e37"))
        edge = QColor(palette.get("accent" if hovered else "border",
                                  "#4c8dff"))
        ink = QColor(palette.get("text", "#e6e6e6"))
        if self.orientation() == Qt.Horizontal:
            w = rect.width()
            h = max(24, w * 3)
            top = rect.center().y() - h // 2
            painter.setPen(edge)
            painter.setBrush(tab)
            painter.drawRoundedRect(0, top, w - 1, h, 3, 3)
            cx, cy, s = rect.center().x(), rect.center().y(), max(2, w // 4)
            points = ([QPoint(cx + s, cy - 2 * s), QPoint(cx - s, cy),
                       QPoint(cx + s, cy + 2 * s)] if towards_start else
                      [QPoint(cx - s, cy - 2 * s), QPoint(cx + s, cy),
                       QPoint(cx - s, cy + 2 * s)])
        else:
            h = rect.height()
            w = max(24, h * 3)
            left = rect.center().x() - w // 2
            painter.setPen(edge)
            painter.setBrush(tab)
            painter.drawRoundedRect(left, 0, w, h - 1, 3, 3)
            cx, cy, s = rect.center().x(), rect.center().y(), max(2, h // 4)
            points = ([QPoint(cx - 2 * s, cy + s), QPoint(cx, cy - s),
                       QPoint(cx + 2 * s, cy + s)] if towards_start else
                      [QPoint(cx - 2 * s, cy - s), QPoint(cx, cy + s),
                       QPoint(cx + 2 * s, cy - s)])
        painter.setPen(Qt.NoPen)
        painter.setBrush(ink)
        painter.drawPolygon(QPolygon(points))

    def _index(self) -> int:
        """This handle's index in its splitter."""
        splitter = self.splitter()
        for i in range(splitter.count()):
            if splitter.handle(i) is self:
                return i
        return -1


class CollapsibleSplitter(QSplitter):
    """A splitter whose panes resize by dragging and collapse at the limit.

    See the module docstring for the pane kinds and the API.

    :param orientation: ``Qt.Vertical`` stacks the panes.
    :param parent: parent widget.
    :param persist_key: where the dragged sizes are remembered, e.g.
        ``"mask::shell"``; empty remembers nothing (what a test wants).
    """

    #: ``(pane name, collapsed, by_user)`` after a pane collapses or opens.
    pane_toggled = Signal(str, bool, bool)

    def __init__(self, orientation, parent=None, *, persist_key: str = ""):
        """Build an empty splitter; add panes with :meth:`add_pane`."""
        super().__init__(orientation, parent)
        self._panes: List[Pane] = []
        self._persist_key = str(persist_key or "")
        self._stored = get_pane_extents(self._persist_key)
        self._laid_out = False
        self._rebalance_queued = False
        self.setHandleWidth(GRIP_PX)
        self.splitterMoved.connect(self._on_moved)

    def createHandle(self):                                  # noqa: N802
        """Every handle is a :class:`_PaneHandle`."""
        return _PaneHandle(self.orientation(), self)

    def _along(self, size) -> int:
        """The part of ``size`` that runs along the splitter."""
        return (size.height() if self.orientation() == Qt.Vertical
                else size.width())

    def add_pane(self, widget: QWidget, name: str, *, folder=None,
                 mode: Optional[str] = None, stretch: int = 1,
                 extent: int = 0, minimum: Optional[int] = None,
                 focus: bool = False, fold_key: str = "",
                 index: Optional[int] = None) -> Pane:
        """Register ``widget`` as the pane ``name``, adding it if need be.

        :param widget: the pane. Added at ``index`` (or the end) unless it is
            already a child of this splitter.
        :param name: stable English name, used for the stored size and the
            handle's tooltip; translated where it is shown.
        :param folder: the heading's Folder; makes it a HEADER pane.
        :param mode: :data:`EDGE` for a pane that collapses to nothing.
        :param stretch: 0 keeps its own size as the splitter grows.
        :param extent: the size to open at when the user never dragged it.
        :param minimum: its minimum while open; ``None`` keeps the widget's.
        :param focus: showing the widget also opens the pane.
        :param fold_key: remembers an EDGE pane's collapse across restarts.
        :returns: the :class:`Pane`.
        """
        if self.indexOf(widget) < 0:
            if index is None:
                super().addWidget(widget)
            else:
                super().insertWidget(index, widget)
        existing = self.pane(name)
        if existing is not None:
            self._panes.remove(existing)
        if mode is None and folder is not None:
            mode = HEADER
        if minimum is None:
            minimum = self._along(widget.minimumSize())
        pane = Pane(widget, name, mode, folder, stretch,
                    self._stored.get(str(name), extent), minimum, focus,
                    fold_key)
        self._panes.append(pane)
        self.setStretchFactor(self.indexOf(widget), max(0, int(stretch)))
        widget.installEventFilter(self)
        if folder is not None:
            folder.add_listener(partial(self._folder_moved, pane))
            lock_folded_to_bottom(widget, folder, self.orientation())
            if folder.shut:
                self._shape(pane, True)
        if mode == EDGE and pane.fold_key:
            try:
                from ..preferences import get_folded_panels

                pane.edge_collapsed = bool(
                    get_folded_panels().get(pane.fold_key))
            except Exception:                                # noqa: BLE001
                LOG.debug("could not read the folded panes", exc_info=True)
        self._sync_flags()
        self._queue_rebalance()
        return pane

    def pane(self, name: str) -> Optional[Pane]:
        """The pane called ``name``, or None.

        :param name: the name it was added under.
        """
        for pane in self._panes:
            if pane.name == str(name):
                return pane
        return None

    def panes(self) -> list:
        """Every registered pane, in the order registered."""
        return list(self._panes)

    def _pane_of(self, widget) -> Optional[Pane]:
        """The pane whose widget is ``widget``."""
        for pane in self._panes:
            if pane.widget is widget:
                return pane
        return None

    def is_collapsed(self, name: str) -> bool:
        """Whether the pane ``name`` is collapsed; False for no such pane.

        :param name: the pane's name.
        """
        pane = self.pane(name)
        return bool(pane is not None and pane.is_collapsed())

    def set_collapsed(self, name: str, collapsed: bool, *,
                      by_user: bool = False) -> bool:
        """Collapse or open the pane ``name``.

        :param name: the pane's name.
        :param collapsed: True to collapse it.
        :param by_user: True when a person asked for it. Only such a change
            is remembered, and only such a change pins a pane against
            :class:`FocusCollapse`.
        :returns: whether the pane is now in the state asked for.
        """
        pane = self.pane(name)
        if pane is None or pane.mode is None:
            return False
        collapsed = bool(collapsed)
        if pane.is_collapsed() == collapsed:
            return True
        if pane.mode == HEADER:
            pane.folder.set_shut(collapsed, by_user=by_user)
        else:
            self._set_edge(pane, collapsed, by_user=by_user)
        return pane.is_collapsed() == collapsed

    def toggle_pane(self, name: str, *, by_user: bool = True) -> bool:
        """Flip the pane ``name``; returns whether it is now collapsed.

        :param name: the pane's name.
        :param by_user: whether a person asked for it; see
            :meth:`set_collapsed`.
        """
        pane = self.pane(name)
        if pane is None:
            return False
        self.set_collapsed(name, not pane.is_collapsed(), by_user=by_user)
        return pane.is_collapsed()

    def _folder_moved(self, pane: Pane, shut: bool, by_user: bool) -> None:
        """A HEADER pane's heading was clicked or folded for the user."""
        if pane not in self._panes or not _alive(pane.widget):
            return
        self._shape(pane, shut)
        self._sync_flags()
        self.rebalance(grow=None if shut else pane)
        self.pane_toggled.emit(pane.name, bool(shut), bool(by_user))

    def _shape(self, pane: Pane, collapsed: bool) -> None:
        """Let a HEADER pane shrink to its heading, or hold its minimum."""
        widget = pane.widget
        index = self.indexOf(widget)
        if collapsed:
            if 0 <= index < self.count():
                size = self.sizes()[index]
                if size > self._collapsed_extent(pane) + GRIP_PX:
                    pane.extent = size
            if self.orientation() == Qt.Vertical:
                widget.setMinimumHeight(0)
            else:
                widget.setMinimumWidth(0)
        else:
            if self.orientation() == Qt.Vertical:
                widget.setMaximumHeight(UNLIMITED)
                widget.setMinimumHeight(int(pane.minimum or 0))
            else:
                widget.setMaximumWidth(UNLIMITED)
                widget.setMinimumWidth(int(pane.minimum or 0))

    def _set_edge(self, pane: Pane, collapsed: bool, *, by_user: bool,
                  from_drag: bool = False) -> None:
        """Collapse or open an EDGE pane."""
        if pane.edge_collapsed == bool(collapsed):
            return
        index = self.indexOf(pane.widget)
        if collapsed and 0 <= index < self.count():
            size = self.sizes()[index]
            if size > 0:
                pane.extent = size
        pane.edge_collapsed = bool(collapsed)
        if by_user and pane.fold_key:
            try:
                from ..preferences import set_folded_panel

                set_folded_panel(pane.fold_key, bool(collapsed))
            except Exception:                                # noqa: BLE001
                LOG.debug("could not store the folded pane", exc_info=True)
        self._sync_flags()
        if not from_drag:
            self.rebalance(grow=None if collapsed else pane)
        self.pane_toggled.emit(pane.name, bool(collapsed), bool(by_user))

    def _collapsed_extent(self, pane: Pane) -> int:
        """How much room a collapsed pane needs: its heading, or nothing."""
        if pane.mode == EDGE:
            return 0
        widget = pane.widget
        layout = widget.layout()
        if layout is not None:
            layout.activate()
        return max(self._along(widget.minimumSizeHint()),
                   self._along(widget.sizeHint()) if pane.is_collapsed()
                   else 0)

    def _sync_flags(self) -> None:
        """Which children Qt may take to zero, and the handles' dress.

        An open pane with a way back is collapsible (a drag past its minimum
        collapses it); a collapsed HEADER pane is not (its heading must stay);
        an EDGE pane always is (zero is its collapsed size); a plain child is
        not.
        """
        for i in range(self.count()):
            pane = self._pane_of(self.widget(i))
            flag = bool(pane is not None and (
                pane.mode == EDGE
                or (pane.mode == HEADER and not pane.is_collapsed())))
            self.setCollapsible(i, flag)
        self._dress_handles()

    def _edge_pane_beside(self, handle) -> Optional[Pane]:
        """The EDGE pane that ``handle`` belongs to, if any."""
        at = -1
        for i in range(self.count()):
            if self.handle(i) is handle:
                at = i
                break
        if at < 0:
            return None
        for pane in self._panes:
            if pane.mode != EDGE:
                continue
            k = self.indexOf(pane.widget)
            if k < 0:
                continue
            if at == k + 1 or (k == self.count() - 1 and at == k and k > 0):
                return pane
        return None

    def _dress_handles(self) -> None:
        """Widen and label the handles that are an EDGE pane's strip."""
        wide = any(p.mode == EDGE for p in self._panes)
        self.setHandleWidth(EDGE_GRIP_PX if wide else GRIP_PX)
        for i in range(self.count()):
            handle = self.handle(i)
            if isinstance(handle, _PaneHandle):
                handle.retranslate_dynamic_content()
                if handle.edge_pane() is not None:
                    handle.setCursor(Qt.SplitHCursor
                                     if self.orientation() == Qt.Horizontal
                                     else Qt.SplitVCursor)
                handle.update()

    def insertWidget(self, index: int, widget: QWidget) -> None:  # noqa: N802
        """Insert as Qt does, and let a wrapping container inherit a pane.

        :param index: where to insert it.
        :param widget: the child to insert.
        """
        super().insertWidget(index, widget)
        self._rebind(widget)

    def addWidget(self, widget: QWidget) -> None:            # noqa: N802
        """Add as Qt does, and let a wrapping container inherit a pane.

        :param widget: the child to add.
        """
        super().addWidget(widget)
        self._rebind(widget)

    def _rebind(self, widget: QWidget) -> None:
        """Hand a pane to the container that now holds its old widget.

        The settings search strip replaces the settings column with a
        container holding the strip and the column. Without this the column
        would silently stop collapsing, and the container would be a plain,
        unnamed child.
        """
        for pane in self._panes:
            old = pane.widget
            if old is widget or not _alive(old):
                continue
            if self.indexOf(old) < 0 and widget.isAncestorOf(old):
                old.removeEventFilter(self)
                pane.widget = widget
                widget.installEventFilter(self)
                self.setStretchFactor(self.indexOf(widget),
                                      max(0, pane.stretch))
        self._sync_flags()
        self._queue_rebalance()

    def eventFilter(self, watched, event) -> bool:           # noqa: N802
        """Follow a pane being shown or hidden by its owner.

        :param watched: a pane's widget.
        :param event: the event; only show and hide are read.
        :returns: False, so the event goes on as usual.
        """
        kind = event.type()
        if kind in (QEvent.ShowToParent, QEvent.HideToParent):
            pane = self._pane_of(watched)
            if pane is not None:
                if (kind == QEvent.ShowToParent and pane.focus
                        and pane.is_collapsed()):
                    self.set_collapsed(pane.name, False, by_user=False)
                self._queue_rebalance()
        return False

    def showEvent(self, event) -> None:                      # noqa: N802
        """Lay the panes out from their remembered sizes, once.

        :param event: the show event.
        """
        super().showEvent(event)
        if not self._laid_out:
            self._laid_out = True
            self._queue_rebalance(first=True)

    def _queue_rebalance(self, first: bool = False) -> None:
        """Rebalance on the next turn of the event loop, once."""
        self._apply_limits()
        if first:
            self._first_layout = True
        if self._rebalance_queued:
            return
        self._rebalance_queued = True
        QTimer.singleShot(0, self._run_queued_rebalance)

    def _run_queued_rebalance(self) -> None:
        """The queued rebalance."""
        self._rebalance_queued = False
        if not _alive(self):
            return
        first = bool(getattr(self, "_first_layout", False))
        self._first_layout = False
        self.rebalance(fresh=first)

    def _visible_indices(self) -> list:
        """Indices of the children that are not hidden."""
        return [i for i in range(self.count())
                if not self.widget(i).isHidden()]

    def _absorber(self, visible) -> Optional[int]:
        """The collapsed pane that takes the spare room, when nothing can.

        Spare room belongs to an open pane that stretches. When there is
        none -- the console folded with only System and the buttons open
        beneath it -- the FIRST collapsed heading takes it, locked to the
        bottom of that room, so every heading and every fixed-size pane
        below it ends up stacked at the bottom of the container. None while
        a stretching pane is open.
        """
        first = None
        for i in visible:
            pane = self._pane_of(self.widget(i))
            if pane is None:
                return None
            if not pane.is_collapsed():
                if pane.stretch > 0:
                    return None
                continue
            if pane.mode == HEADER and first is None:
                first = i
        return first

    def _apply_limits(self) -> None:
        """Hold every collapsed HEADER pane at its heading's size.

        A maximum rather than a size: Qt keeps it through every window
        resize and every drag, where a size would be handed the growth.
        """
        visible = self._visible_indices()
        absorber = self._absorber(visible)
        vertical = self.orientation() == Qt.Vertical
        for i in range(self.count()):
            pane = self._pane_of(self.widget(i))
            if pane is None:
                continue
            widget = pane.widget
            if pane.mode != HEADER:
                continue
            if pane.is_collapsed() and i != absorber:
                limit = self._collapsed_extent(pane)
            else:
                limit = UNLIMITED
            if vertical:
                widget.setMaximumHeight(limit)
            else:
                widget.setMaximumWidth(limit)

    def _height_for_width(self, widget) -> int:
        """The height ``widget`` needs at this splitter's width, or 0.

        A splitter ignores height-for-width, which a row of buttons that
        wraps onto a second line depends on: given its one-line size hint, the
        second line was simply cut off. The need is read here and used as the
        pane's size when it is laid out, and again when the width changes.
        NOT as a minimum: the splitter's minimum is the window's, and a row
        that wraps at a narrow first width would hold a laptop window taller
        than its screen.
        """
        if self.orientation() != Qt.Vertical or not widget.hasHeightForWidth():
            return 0
        width = self.width()
        if width <= 0:
            return 0
        try:
            return max(0, int(widget.heightForWidth(width)))
        except Exception:                                    # noqa: BLE001
            return 0

    def resizeEvent(self, event) -> None:                    # noqa: N802
        """A new width can wrap a row onto another line; make room for it.

        A resize of the window is not a drag -- no handle is held -- so
        re-fitting here fights nobody.

        :param event: the resize event.
        """
        super().resizeEvent(event)
        if (self._laid_out and self.orientation() == Qt.Vertical
                and event.oldSize().width() != event.size().width()
                and not getattr(self, "_refit_queued", False)):
            self._refit_queued = True
            QTimer.singleShot(0, self._refit)

    def _refit(self) -> None:
        """The queued re-fit of the fixed-size panes that wrap."""
        self._refit_queued = False
        if _alive(self):
            self.rebalance(refit=True)

    def rebalance(self, grow: Optional[Pane] = None,
                  fresh: bool = False, refit: bool = False) -> list:
        """Give every visible pane its share and return the new sizes.

        Collapsed HEADER panes get their heading, collapsed EDGE panes get
        nothing, panes with ``stretch`` 0 get their own size, and the rest of
        the room is shared by the others in proportion to their current size.
        A pane being opened (``grow``), one too small to have a real size, and
        every pane on the first layout (``fresh``) are sized from their
        remembered size instead. ``refit`` sizes a fixed-size pane that wraps
        to the height its width now needs.

        :returns: the sizes set, or ``[]`` when there was nothing to lay out.
        """
        self._apply_limits()
        count = self.count()
        visible = self._visible_indices()
        if not visible:
            return []
        sizes = list(self.sizes())
        handles = self.handleWidth() * max(0, len(visible) - 1)
        room = self._along(self.size()) - handles
        total = sum(sizes[i] for i in visible)
        if room > 0 and self._laid_out:
            total = room
        absorber = self._absorber(visible)
        fixed = {}
        rigid = {}
        elastic = {}
        for i in visible:
            widget = self.widget(i)
            pane = self._pane_of(widget)
            if pane is not None and pane.is_collapsed():
                fixed[i] = (0 if pane.mode == EDGE
                            else self._collapsed_extent(pane))
                continue
            natural = self._along(widget.sizeHint())
            current = sizes[i]
            remembered = pane.extent if pane is not None else 0
            small = current <= GRIP_PX * 4
            if pane is not None and (pane is grow or small or fresh):
                want = remembered or natural or current
            else:
                want = current or remembered or natural
            stretch = pane.stretch if pane is not None else 1
            wraps = self._height_for_width(widget)
            if pane is not None and pane is grow and remembered:
                rigid[i] = remembered
            elif stretch <= 0 and wraps and (refit or fresh or small):
                rigid[i] = max(wraps, self._along(widget.minimumSizeHint()))
            elif stretch <= 0:
                rigid[i] = max(want, natural if fresh else 0)
            else:
                elastic[i] = max(1, want)
        if total <= 0:
            total = (sum(fixed.values()) + sum(rigid.values())
                     + sum(elastic.values()))
        if absorber is not None:
            others = (sum(v for j, v in fixed.items() if j != absorber)
                      + sum(rigid.values()))
            fixed[absorber] = max(fixed[absorber], total - others)
        spare = total - sum(fixed.values()) - sum(rigid.values())
        if spare < 0 and rigid:
            scale = max(0.0, (total - sum(fixed.values()))
                        / float(sum(rigid.values()) + sum(elastic.values())))
            rigid = {i: int(v * scale) for i, v in rigid.items()}
            elastic = {i: max(1, int(v * scale)) for i, v in elastic.items()}
            spare = sum(elastic.values())
        new = list(sizes)
        for i, v in fixed.items():
            new[i] = int(v)
        for i, v in rigid.items():
            new[i] = int(v)
        weight = float(sum(elastic.values())) or 1.0
        for i, v in elastic.items():
            new[i] = max(0, int(max(0, spare) * v / weight))
        for i in range(count):
            if i not in visible:
                new[i] = sizes[i]
        self.setSizes(new)
        return new

    def _on_moved(self, _pos: int, _index: int) -> None:
        """The user dragged a handle: collapse at the limit, remember sizes.

        Never rebalances. A drag is the user placing the handle, and moving
        anything they did not move while they hold it is the fight item 471
        rules out.
        """
        sizes = self.sizes()
        for i, size in enumerate(sizes):
            if i >= self.count():
                break
            widget = self.widget(i)
            pane = self._pane_of(widget)
            if pane is None or widget.isHidden():
                continue
            if pane.mode == EDGE:
                if size == 0 and not pane.edge_collapsed:
                    self._set_edge(pane, True, by_user=True, from_drag=True)
                elif size > 0 and pane.edge_collapsed:
                    self._set_edge(pane, False, by_user=True, from_drag=True)
            elif (pane.mode == HEADER and size == 0
                  and not pane.is_collapsed()):
                pane.folder.set_shut(True, by_user=True)
                continue
            if not pane.is_collapsed() and size > 0:
                pane.extent = size
        self._save_extents()

    def extents(self) -> dict:
        """``{pane name: size to open at}`` for every registered pane."""
        return {p.name: p.extent for p in self._panes if p.extent > 0}

    def _save_extents(self) -> None:
        """Store the dragged sizes under this splitter's key."""
        if self._persist_key:
            set_pane_extents(self._persist_key, self.extents())


class FocusCollapse(QObject):
    """Collapse the surroundings while a focus widget is on screen.

    Item 471: "when a live preview opens, it takes the whole screen height:
    the console, the System container and the button section below System
    auto-collapse, and the settings collapse to the left", and the same
    whenever figures appear.

    FOCUS widgets are watched (:meth:`watch`); while ANY of them is shown the
    TARGET panes (:meth:`target`) are collapsed, and when the last is hidden
    the targets this collapsed are opened again. A target that was already
    collapsed -- by the user, or remembered from last session -- is left
    exactly as it is, both ways.

    NEVER FIGHTS THE USER. A target the user opens by hand (clicking its
    heading or its handle, or dragging it open) is PINNED: nothing here
    collapses it again until the view ends. A target the user collapses by
    hand is theirs too, and is not re-opened when the focus goes.

    A VIEW is one visit to one screen: it begins when the screen is shown
    (:meth:`begin_view`, from its ``showEvent``) and ends when it is hidden
    because the user went to another screen (:meth:`end_view`, from its
    ``hideEvent``). Minimising the window does not end it -- the owner
    passes only non-spontaneous events. Pins last for the view: returning to
    the screen with its live preview still open collapses the surroundings
    again, which is what a fresh arrival at that layout would get.

    :param parent: owner, normally the screen.
    """

    def __init__(self, parent=None):
        """Watch nothing and collapse nothing until told what to."""
        super().__init__(parent)
        self._triggers: list = []
        self._targets: list = []
        self._splitters: list = []
        self._pinned: set = set()
        self._auto: set = set()
        self._active = False

    def watch(self, widget: QWidget) -> None:
        """Treat ``widget`` being shown as a reason to collapse the targets.

        Only its hidden flag is read -- it is never shown, measured or
        built, so a lazily-built panel stays unbuilt until its owner shows
        it.

        :param widget: the focus widget.
        """
        if widget is None or widget in self._triggers:
            return
        self._triggers.append(widget)
        widget.installEventFilter(self)
        self.refresh()

    def target(self, splitter: CollapsibleSplitter, name: str) -> None:
        """Collapse the pane ``name`` of ``splitter`` while a focus is shown.

        :param splitter: the splitter holding the pane.
        :param name: the pane's name.
        """
        if splitter is None or splitter.pane(name) is None:
            return
        if (splitter, name) in self._targets:
            return
        self._targets.append((splitter, str(name)))
        if splitter not in self._splitters:
            self._splitters.append(splitter)
            splitter.pane_toggled.connect(
                partial(self._on_toggled, splitter))
        if self._active:
            self._collapse_one(splitter, str(name))

    @staticmethod
    def _key(splitter, name) -> tuple:
        """How a target is recorded."""
        return (id(splitter), str(name))

    def is_pinned(self, splitter, name) -> bool:
        """Whether the user opened ``name`` by hand during this view.

        :param splitter: the splitter holding the pane.
        :param name: the pane's name.
        """
        return self._key(splitter, name) in self._pinned

    def _on_toggled(self, splitter, name, collapsed, by_user) -> None:
        """Hear the user's own folds, which outrank this class's.

        Opening a target is a pin only while a focus widget is shown: that is
        the user overruling the layout this class made. Opening the console
        before any preview exists overrules nothing, and must not stop the
        preview, opened later, from taking the height.
        """
        if not by_user:
            return
        key = self._key(splitter, name)
        self._auto.discard(key)
        if collapsed:
            self._pinned.discard(key)
        elif self.is_active():
            self._pinned.add(key)

    def is_active(self) -> bool:
        """Whether any focus widget is shown."""
        live = []
        shown = False
        for widget in self._triggers:
            if not _alive(widget):
                continue
            live.append(widget)
            if not widget.isHidden():
                shown = True
        self._triggers = live
        return shown

    def eventFilter(self, watched, event) -> bool:           # noqa: N802
        """A focus widget was shown or hidden.

        :param watched: the focus widget.
        :param event: the event; only show and hide are read.
        :returns: False, so the event goes on as usual.
        """
        if event.type() in (QEvent.ShowToParent, QEvent.HideToParent):
            self.refresh()
        return False

    def refresh(self, *, force: bool = False) -> bool:
        """Collapse or restore the targets for what is shown now.

        Acts on a CHANGE (or when ``force``), so a figure arriving while a
        preview is already open does not re-collapse what the user opened.

        :returns: whether a focus widget is shown.
        """
        now = self.is_active()
        if now and (force or not self._active):
            for splitter, name in list(self._targets):
                self._collapse_one(splitter, name)
        elif not now and (force or self._active):
            self._release()
        self._active = now
        return now

    def _collapse_one(self, splitter, name) -> None:
        """Collapse one target unless the user pinned it or already did."""
        key = self._key(splitter, name)
        if key in self._pinned or not _alive(splitter):
            return
        if splitter.is_collapsed(name):
            return
        if splitter.set_collapsed(name, True, by_user=False):
            self._auto.add(key)

    def _release(self) -> None:
        """Open what this class collapsed and the user has not touched."""
        for splitter, name in list(self._targets):
            key = self._key(splitter, name)
            if key not in self._auto:
                continue
            if _alive(splitter) and splitter.is_collapsed(name):
                splitter.set_collapsed(name, False, by_user=False)
        self._auto.clear()

    def begin_view(self) -> None:
        """The screen came on: forget last view's pins and apply the rule."""
        self._pinned.clear()
        self.refresh(force=True)

    def end_view(self) -> None:
        """The user left the screen: their pins for this view are spent."""
        self._pinned.clear()


def splitter_of(widget) -> Optional[CollapsibleSplitter]:
    """The CollapsibleSplitter ``widget`` sits directly in, if any.

    :param widget: any widget.
    """
    parent = widget.parentWidget() if _alive(widget) else None
    return parent if isinstance(parent, CollapsibleSplitter) else None


__all__ = [
    "CollapsibleSplitter", "EDGE", "FocusCollapse", "HEADER", "Pane",
    "get_pane_extents", "lock_folded_to_bottom", "set_pane_extents",
    "splitter_of",
]
