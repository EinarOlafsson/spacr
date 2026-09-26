"""The left navigation dock: an icon, a name, and a category heading.

A row is a button with an icon and its name, always both. The only thing the
pointer changes is the colour, and the explanation goes to the strip along
the bottom of the window rather than into a popup.

WHAT WAS REMOVED, AND WHY EACH ONE WAS THE BUG. The dock this replaces was
1,116 lines across ``Sidebar`` and ``_DockRow`` in ``spacr.qt.app``, and
nearly all of it was machinery that existed to defeat itself:

* a translucent slab painted in ``paintEvent`` — the "black box" of four
  separate commits, which turned out to be the dock painting itself rather
  than any stylesheet;
* a per-row icon-size model (``resting_icon_px``, ``_place_icon``,
  ``_set_icon_px``, ``_forget_icon_sizes``, ``_rest_every_icon``) that grew
  and shrank icons under the pointer, which is what made hovering relayout
  the column and blink;
* the name painted only while hovered, so a resting dock was a column of
  unlabelled glyphs;
* a second, indented level of folded modules with its own expand state
  (``_fold_children``, ``_open_hosts``), which is the "sub categories".

None of that is here. A row is a button with an icon and its name, always
both. The only thing the pointer changes is the colour.

WHERE THE EXPLANATION WENT. Not into a popup tooltip — those are explicitly
unwanted — but into the strip along the bottom of the window, which already
exists as :mod:`spacr.qt.widgets.module_hint_bar` and already holds the last
hovered module for thirty seconds with its API and tutorial links. This dock
only says which module is under the pointer, via :attr:`Dock.module_hovered`;
the bar decides how to explain it.

WHAT IS KEPT, BECAUSE SOMETHING ELSE READS IT. Categories still collapse —
the list is longer than a short screen. Rows still carry ``navKey`` and
headers are still ``SidebarSection``, because the theme, the tutorial script
and the maturity tests all reach the dock through those names. And
:meth:`refresh_visibility` still applies the Alpha/Beta maturity filter and
hides a heading whose modules are all filtered out, which is a separate
reason for a row to be absent from its section being shut.

NOTHING HERE IMPORTS :mod:`spacr.qt.app`. The registry lives there and would
be a circular import, so the rows, the icon lookup and the maturity
predicate are all injected.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Tuple

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget

from ..i18n import tr
from ..theme import active_palette
from .eliding import ElidingPushButton

#: A dock row as the registry hands it over: key, name, description, section.
Row = Tuple[str, str, str, str]

#: The key of the row that goes Home. Never filtered out.
HOME_KEY = "__home__"

#: Icon edge, in unscaled pixels. Big enough to read as a picture rather
#: than the bullet the dock used to draw, and the same in every state.
ICON_PX = 20

#: Space between the dock's edge and its rounded panel, in pixels.
#:
#: WITHOUT IT THE CORNERS ARE NOT VISIBLE. A rounded rectangle flush against
#: the window edge has its curve cut off by the edge it is flush with, which
#: is the same shape as no rounding at all.
PANEL_INSET = 6

#: The panel's corner radius. HomePanelBox's number, because the request was
#: for the dock to look like that box and not merely to be rounded.
PANEL_RADIUS = 8


class DockRow(ElidingPushButton):
    """One module: its icon, then its name, both always drawn.

    The row paints nothing of its own — the colour comes from the
    stylesheet :class:`Dock` installs, so there is one place that decides
    what hover looks like and no ``paintEvent`` to disagree with it.
    """

    hovered = Signal(str, bool)          #: key, and whether the pointer entered

    def __init__(self, key: str, name: str, desc: str = "", parent=None):
        """Build one module row.

        :param key: the module's registry key. Stamped onto the row three
            times over -- as ``navKey``, as ``moduleAppKey`` and as the
            attribute -- because three different readers ask for it: the
            icon refresh, the bottom hint strip's filter, and this module.
        :param name: the module's name, drawn beside the icon and set as
            the accessible name so a screen reader still gets the whole of
            it when the column elides it.
        :param desc: the one-line summary. Not drawn here at all: it is
            stamped as ``moduleSummarySource`` for the strip along the
            bottom of the window, which is where descriptions go.
        :param parent: parent widget.
        """
        super().__init__(name.replace("&", "&&"), parent)
        self.key = key
        self.desc = desc
        self.setObjectName("SidebarItem")
        self.setProperty("moduleNameSource", name)
        self.setProperty("moduleSummarySource", desc)
        self.setProperty("navKey", key)
        self.setProperty("moduleAppKey", key)
        self.setProperty("moduleTooltipStyle", "sidebar")
        self.setAccessibleName(name)
        self.setAccessibleDescription(desc)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Preferred,
                           QSizePolicy.Policy.Fixed)
        self.setToolTip("")
        self._hovered = False

    def is_hovered(self) -> bool:
        """Whether the pointer is currently on this row."""
        return self._hovered

    def enterEvent(self, event):                # noqa: N802 - Qt naming
        """Light the row as the pointer arrives.

        :param event: the Qt enter event.
        """
        self._hovered = True
        self.hovered.emit(self.key, True)
        super().enterEvent(event)

    def leaveEvent(self, event):                # noqa: N802 - Qt naming
        """Drop the highlight as the pointer leaves.

        :param event: the Qt leave event.
        """
        self._hovered = False
        self.hovered.emit(self.key, False)
        super().leaveEvent(event)


class SectionHeader(QLabel):
    """A category heading. A label rather than a button, because it is
    already styled as a heading and a button would have to be un-styled
    back into one; the click arrives through :meth:`Dock.eventFilter`.

    :param section: the category name. Shown as the heading AND kept on the
        ``sectionName`` property, which is how the dock finds the rows a
        click should fold -- the visible text is translated, the property
        is not.
    :param parent: parent widget; ownership only.
    """

    def __init__(self, section: str, parent=None):
        """Build one dock section heading.

        The section name is also stored under its legacy property name, which is
        what the theme styles and what the maturity test looks headers up by.

        :param section: the section's name.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(section, parent)
        self.section = section
        self.setObjectName("SidebarSection")
        self.setProperty("sectionName", section)
        self.setCursor(Qt.CursorShape.PointingHandCursor)


class Dock(QWidget):
    """The navigation column: categories, each holding icon+name rows.

    :param rows: the modules to draw, in order, as ``(key, name, desc,
        section)``. Grouping IS ordering: a new heading starts whenever the
        section changes, so a row out of place draws its heading twice.
    :param icon_for: optional ``key -> QIcon | None`` for the row icons.
    :param is_visible: optional ``key -> bool`` maturity predicate. Injected
        rather than imported so this module does not depend on
        :mod:`spacr.qt.app`, which is what defines the registry.
    :param parent: parent widget; ownership only.
    """

    nav_selected = Signal(str)           #: a row was clicked
    module_hovered = Signal(str)         #: a row is under the pointer

    #: The column starts at ``WIDTH_MIN`` and widens, up to ``WIDTH_MAX``,
    #: if the longest name needs it. Both scale with the font.
    WIDTH_MIN = 220
    WIDTH_MAX = 320

    #: How far the user may drag the column's edge (item 529), before the
    #: font scale. Narrower than ``WIDTH_MIN`` is allowed: a long name elides
    #: rather than pushing the page.
    DRAG_MIN = 150
    DRAG_MAX = 520

    def __init__(self, rows: Iterable[Row],
                 icon_for: Optional[Callable[[str], object]] = None,
                 is_visible: Optional[Callable[[str], bool]] = None,
                 parent=None):
        """Build the dock column: section headings with their module rows under them.

        The container is made transparent rather than coloured. The application
        sheet carries a blanket ``QWidget { background-color: bg }``, so any
        untagged container paints an opaque rectangle -- and a plain widget
        holding a rounded panel is exactly that, a square of window colour
        behind rounded corners. Colouring it only changes which colour the
        rectangle is.

        The rounded panel is a child frame rather than this widget's own
        background, which satisfies both constraints at once: the container
        stays transparent where the theme pins it transparent, and the panel is
        still drawn.

        :param rows: the module rows to list, in order.
        :param icon_for: called with a key for that module's icon.
        :param is_visible: called with a key to decide whether to list it.
        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setObjectName("Dock")
        from ..theme import make_transparent
        make_transparent(self)
        self._icon_for = icon_for
        self._is_visible = is_visible
        self._rows: List[DockRow] = []
        self._headers: Dict[str, SectionHeader] = {}
        self._section_rows: Dict[str, List[DockRow]] = {}
        self._section_of: Dict[str, str] = {}
        self._open: set = set()
        self._items = self._rows
        self._section_headers = self._headers

        outer = QVBoxLayout(self)
        outer.setContentsMargins(PANEL_INSET, PANEL_INSET,
                                 PANEL_INSET, PANEL_INSET)
        outer.setSpacing(0)

        self._panel = QFrame(self)
        self._panel.setObjectName("DockPanel")
        outer.addWidget(self._panel)
        panel_column = QVBoxLayout(self._panel)
        panel_column.setContentsMargins(0, 0, 0, 0)
        panel_column.setSpacing(0)

        title = QLabel("spaCR")
        title.setObjectName("SidebarTitle")
        panel_column.addWidget(title)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("SidebarScroll")
        make_transparent(self._scroll)
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.viewport().setAutoFillBackground(False)

        inner = QWidget()
        inner.setObjectName("SidebarInner")
        column = QVBoxLayout(inner)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        self._scroll.setWidget(inner)
        panel_column.addWidget(self._scroll, 1)

        current = None
        for key, name, desc, section in rows:
            if section and section != current:
                header = SectionHeader(section)
                header.installEventFilter(self)
                column.addWidget(header)
                self._headers[section] = header
                self._section_rows.setdefault(section, [])
                if not self._headers or len(self._headers) == 1:
                    self._open.add(section)
                current = section
            row = DockRow(key, name, desc)
            row.clicked.connect(lambda _checked=False, k=key:
                                self.nav_selected.emit(k))
            row.hovered.connect(self._on_row_hovered)
            column.addWidget(row)
            self._rows.append(row)
            if section:
                self._section_rows.setdefault(section, []).append(row)
            self._section_of[key] = section or ""
        column.addStretch(1)

        self.refresh_icons()
        self.apply_theme()
        self.refresh_visibility()

    def _on_row_hovered(self, key: str, entered: bool) -> None:
        """Name the hovered module, and light exactly that one row.

        Only ENTERING is reported to the strip. A leave that cleared the bar
        would empty it the moment the pointer set off toward the links it
        holds, which is the whole reason that bar keeps its last module.

        THE INK IS NOT `:hover`, and that is a fix rather than a preference.
        The rows were coloured by a `QPushButton#SidebarItem:hover` rule,
        which Qt drives from `WA_UnderMouse` -- and that attribute sticks
        when the widget under the pointer is replaced without the pointer
        moving, which is exactly what clicking a dock row does: the stack
        swaps a whole screen in underneath it and no Leave is ever
        delivered. Reported: "run compare and run history are
        always blue in the dock", and both are rows a user had
        opened. Read off the screen recording: Run History accent-coloured
        while Database Browser and Report above and below it are white and
        the pointer is elsewhere entirely.

        So the dock lights the row itself, from one pass over all of them.
        At most one can be lit, whatever Qt believes about who is under the
        pointer.
        """
        self._light_only(key if entered else None)
        if entered:
            self.module_hovered.emit(key)

    def _light_only(self, key) -> None:
        """Ink the row named by ``key`` and no other. ``None`` clears all.

        Re-polished per row rather than by re-applying the sheet: a
        stylesheet reset re-polishes every widget in the dock, and this runs
        on every pointer move across the column.
        """
        for row in self._rows:
            want = (key is not None and getattr(row, "key", None) == key)
            if bool(row.property("hovered")) == want:
                continue
            row.setProperty("hovered", want)
            style = row.style()
            if style is not None:
                style.unpolish(row)
                style.polish(row)

    def leaveEvent(self, event):                # noqa: N802 - Qt naming
        """The pointer left the column: no row is lit.

        The rows' own Leave covers a pointer stepping between them; this
        covers one that leaves the dock altogether, including straight off
        the bottom row onto the empty stretch below it, where no other row
        will ever be entered.

        :param event: the leave event, passed to the base class after every row
            is unlit.
        """
        self._light_only(None)
        super().leaveEvent(event)

    def eventFilter(self, watched, event):       # noqa: N802 - Qt naming
        """Light a heading under the pointer, and toggle it on release.

        ON RELEASE, NOT PRESS: a press that toggled would fire while the
        pointer was still down, so a drag that began on a heading and ended
        elsewhere would still have shut the section.

        The hover state is a PROPERTY rather than a colour set from here,
        because the stylesheet is the one place that decides what the dock
        looks like.

        :param watched: the object the event is for; only a
            :class:`SectionHeader` is handled here.
        :param event: the event; Enter and Leave set the header's ``hovered``
            property, and a left-button release toggles its section.
        """

        if isinstance(watched, SectionHeader):
            kind = event.type()
            if kind in (QEvent.Type.Enter, QEvent.Type.Leave):
                watched.setProperty("hovered", kind == QEvent.Type.Enter)
                watched.style().unpolish(watched)
                watched.style().polish(watched)
            elif (kind == QEvent.Type.MouseButtonRelease
                    and event.button() == Qt.MouseButton.LeftButton):
                self.toggle_section(watched.section)
                return True
        return super().eventFilter(watched, event)

    def hovered_row(self) -> Optional[DockRow]:
        """The row under the pointer, or ``None``."""
        for row in self._rows:
            if row.is_hovered():
                return row
        return None

    def sync_hover(self, entered=None) -> Optional[str]:
        """Report which row the pointer is on.

        The old dock needed this to repair hover state it had broken by
        relaying out under the pointer. Nothing relayouts now, so this only
        answers the question. Kept because ``tools/diagnose_dock.py`` asks.
        """
        row = self.hovered_row()
        return row.key if row is not None else None

    def sections(self) -> List[str]:
        """Every category heading, in the order they are drawn."""
        return list(self._headers)

    def rows(self) -> List[DockRow]:
        """Every module row, in the order they are drawn."""
        return list(self._rows)

    def section_is_open(self, section: str) -> bool:
        """Whether ``section``'s rows are currently shown.

        :param section: the category name.
        """
        return section in self._open

    def toggle_section(self, section: str) -> bool:
        """Open a closed category or close an open one. Returns the new state.

        :param section: the category name to open or close.
        """
        if section in self._open:
            self._open.discard(section)
        else:
            self._open.add(section)
        self.refresh_visibility()
        return section in self._open

    def expand_host(self, host_key: str) -> None:
        """Accepted and does nothing: there are no folded child rows.

        The second level was removed on request. This remains so the callers
        that opened a host on navigation do not have to know that, and
        because a method that quietly disappeared would fail at the call
        site rather than here, where the reason is written down.

        :param host_key: the app key of the host; ignored.
        """
        return None

    def host_is_expanded(self, host_key: str) -> bool:
        """Always ``False``: there are no folded child rows to expand.

        :param host_key: the app key of the host; ignored.
        """
        return False

    def refresh_visibility(self) -> None:
        """Show a row if its category is open AND maturity allows it.

        Two separate reasons for a row to be absent, and they are kept
        separate: a shut section hides rows that are perfectly mature, and
        the Alpha/Beta filter hides rows inside an open one. A heading
        stays put whether its section is open or shut — it is what you
        click to open it — and hides only when every module beneath it is
        filtered out.
        """
        allowed = self._is_visible or (lambda _key: True)
        populated = set()
        for row in self._rows:
            mature = row.key == HOME_KEY or bool(allowed(row.key))
            section = self._section_of.get(row.key, "")
            row.setVisible(mature and (not section or section in self._open))
            if mature and section:
                populated.add(section)
        for section, header in self._headers.items():
            header.setVisible(section in populated)
            header.setProperty("open", section in self._open)
            header.style().unpolish(header)
            header.style().polish(header)
        self.setFixedWidth(self.column_width())

    def column_width(self) -> int:
        """The width the column wears: the one the user dragged, else fitting.

        Item 529. A width the user chose is kept through every refresh, every
        hide and show, and every session; with none stored the column fits
        its longest name, as :meth:`fitting_width` has always done.
        """
        try:
            from ..preferences import get_dock_width

            wanted = get_dock_width()
        except Exception:                                        # noqa: BLE001
            wanted = 0
        if wanted <= 0:
            return self.fitting_width()
        return self.clamp_width(wanted)

    def clamp_width(self, width: int) -> int:
        """``width`` held between :attr:`DRAG_MIN` and :attr:`DRAG_MAX`.

        :param width: logical pixels.
        """
        from ..preferences import scaled_px

        return max(scaled_px(self.DRAG_MIN),
                   min(int(width), scaled_px(self.DRAG_MAX)))

    def set_column_width(self, width: int, *, remember: bool = True) -> int:
        """Give the column ``width`` within its bounds and maybe remember it.

        :param width: logical pixels; 0 or less goes back to the fitting
            width and forgets the dragged one.
        :param remember: store it for the next session.
        :returns: the width applied.
        """
        if width is None or int(width) <= 0:
            applied = self.fitting_width()
            stored = 0
        else:
            applied = stored = self.clamp_width(width)
        self.setFixedWidth(applied)
        if remember:
            try:
                from ..preferences import set_dock_width

                set_dock_width(stored)
            except Exception:                                    # noqa: BLE001
                pass
        return applied

    def refresh_icons(self) -> None:
        """Re-ask the provider for every row's icon.

        A QIcon bakes its pixmap when it is built, so re-applying the
        stylesheet does not recolour icons that already exist.

        THE SIZE IS FIXED PER SCALE, NOT PER ROW STATE. The old dock grew
        and shrank an icon on hover, which relayed out the whole column
        under the pointer; that is what :data:`ICON_PX` being one number
        stops. It still has to follow the interface scale, so the base is
        recorded on each row and re-derived from there -- see
        :func:`spacr.qt.preferences._set_scaled_icon_size` for why it is
        never recomputed from the size the row is already wearing.
        """
        from ..preferences import _set_scaled_icon_size
        for row in self._rows:
            _set_scaled_icon_size(row, ICON_PX)
            if self._icon_for is None:
                continue
            key = getattr(row, "key", None)
            if key is None:
                continue
            icon = self._icon_for(key)
            if icon is not None:
                row.setIcon(icon)

    def apply_theme(self) -> None:
        """Paint the rounded panel, and the one rule hover uses.

        THE PANEL IS THE ONLY THING THAT PAINTS. Everything inside it is
        transparent on purpose: a title or a heading carrying a fill of its
        own would draw a square corner over the rounded one directly beneath
        it, which is the exact shape this was asked to stop being.

        The three values come from HomePanelBox rather than being chosen
        again here -- ``pane_surface('surface_alt')``, ``border_soft`` and an
        8 px radius -- so the dock and that box stay the same material when
        either is restyled.
        """
        from ..theme import pane_surface

        palette = active_palette()
        accent = palette["accent"]
        self.setStyleSheet(
            "QFrame#DockPanel {"
            f"  background: {pane_surface('surface_alt')};"
            f"  border: 1px solid {palette['border_soft']};"
            f"  border-radius: {PANEL_RADIUS}px;"
            "}"
            "QScrollArea#SidebarScroll, QWidget#SidebarInner {"
            "  background: transparent; border: none;"
            "}"
            "QLabel#SidebarTitle { background: transparent; }"
            "QPushButton#SidebarItem {"
            "  background: transparent; border: none; text-align: left;"
            "  padding: 6px 10px;"
            "}"
            f'QPushButton#SidebarItem[hovered="true"] {{ color: {accent}; }}'
            "QLabel#SidebarSection {"
            "  padding: 10px 10px 4px 10px; font-weight: 600;"
            "  background: transparent;"
            "}"
            f"QLabel#SidebarSection:hover {{ color: {accent}; }}"
        )

    def row_height(self) -> int:
        """The height of a row, or 0 if the dock is empty."""
        return self._rows[0].sizeHint().height() if self._rows else 0

    def fitting_width(self) -> int:
        """Width that shows the longest visible name in full, within bounds.

        Font scale moves both bounds; the widest visible row moves the
        result between them. Public because the locked dock re-applies it
        after being re-parented out of the drawer, which had resized it.
        """
        from PySide6.QtGui import QFontMetrics

        from ..preferences import scaled_px

        widest = 0
        for row in self._rows:
            if row.isHidden():
                continue
            metrics = QFontMetrics(row.font())
            text = getattr(row, "full_text", lambda: row.text())()
            widest = max(widest, metrics.horizontalAdvance(str(text)))
        room = widest + scaled_px(ICON_PX) + scaled_px(40)
        return max(scaled_px(self.WIDTH_MIN),
                   min(room, scaled_px(self.WIDTH_MAX)))

    def clipped_items(self) -> list:
        """Rows whose name had to be shortened to fit.

        Empty in a healthy layout, and a test asserts that.
        """
        return [r for r in self._rows if r.is_elided()]


class DockEdge(QWidget):
    """The strip along the dock's right edge that drags its width (item 529).

    Dragging it sets the width of the dock while the dock is shown. A
    sibling of the dock's slot rather than a
    splitter handle, because the dock is a fixed-width layout member and
    everything that measures it (the drawer, the backdrop, the fitting width)
    reads that fixed width. Dragging sets it; releasing stores it; a
    double-click forgets it and the column fits its names again. It is
    shown and hidden with the dock, so a hidden dock has no edge to catch.

    :param dock: the :class:`Dock` it resizes.
    :param parent: parent widget; ownership only.
    """

    #: The grab area, in pixels: the same as a splitter handle's.
    GRIP_PX = 6

    def __init__(self, dock: "Dock", parent=None):
        """Build the edge for ``dock``; hidden until the dock is shown."""
        super().__init__(parent)
        self._dock = dock
        self._pressed_x = None
        self._start_width = 0
        self.setObjectName("DockEdge")
        self.setStyleSheet(
            "QWidget#DockEdge { background: transparent; border: none; }")
        self.setAttribute(Qt.WA_Hover, True)
        self.setFixedWidth(self.GRIP_PX)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setCursor(Qt.SizeHorCursor)
        self.setAccessibleName(tr("Dock width"))
        self.retranslate_dynamic_content()
        self.hide()

    def retranslate_dynamic_content(self, language=None) -> None:
        """Rewrite the tooltip in ``language``."""
        self.setToolTip(tr("Drag to make the dock wider or narrower. "
                           "Double-click to fit it to the names again.",
                           language))

    def enterEvent(self, event) -> None:                     # noqa: N802
        """Light the line up under the pointer.

        :param event: the event.
        """
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:                     # noqa: N802
        """Put the line back.

        :param event: the event.
        """
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:                # noqa: N802
        """Begin a drag from the dock's present width.

        :param event: the event.
        """
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        self._pressed_x = event.globalPosition().x()
        self._start_width = self._dock.width()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:                 # noqa: N802
        """Follow the pointer, within the dock's bounds; stored on release.

        :param event: the event.
        """
        if self._pressed_x is None:
            super().mouseMoveEvent(event)
            return
        moved = event.globalPosition().x() - self._pressed_x
        self._dock.set_column_width(self._dragged_to(moved),
                                    remember=False)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:              # noqa: N802
        """End the drag and remember the width it left.

        :param event: the event.
        """
        if self._pressed_x is None:
            super().mouseReleaseEvent(event)
            return
        moved = event.globalPosition().x() - self._pressed_x
        self._pressed_x = None
        if moved:
            self._dock.set_column_width(self._dragged_to(moved))
        self.update()
        event.accept()

    def _dragged_to(self, moved: float) -> int:
        """The width a drag of ``moved`` pixels asks for, never below 1.

        Never 0 or less: to :meth:`Dock.set_column_width` that means
        "forget the dragged width", and a drag past the left bound is the
        user asking for the narrowest dock, not the fitting one.
        """
        return max(1, int(round(self._start_width + moved)))

    def mouseDoubleClickEvent(self, event) -> None:          # noqa: N802
        """Forget the dragged width; the column fits its names again.

        :param event: the event.
        """
        self._pressed_x = None
        self._dock.set_column_width(0)
        self.update()
        event.accept()

    def paintEvent(self, _event) -> None:                    # noqa: N802
        """Draw the one-pixel line a splitter handle draws.

        :param _event: the paint event; the whole strip is drawn.
        """
        palette = active_palette() or {}
        hovered = self.underMouse() or self._pressed_x is not None
        line = QColor(palette.get("accent" if hovered else "border_soft",
                                  "#4c8dff" if hovered else "#3a3f4b"))
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(rect.center().x(), rect.top(), 1, rect.height(), line)
        painter.end()
