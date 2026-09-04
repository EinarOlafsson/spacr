"""Display saved and interactive run figures in a scrollable grid.

Saved figures are placed in uniform grid cells and rendered at their original
aspect ratios, preserving the geometry of views such as plate heatmaps. The
number of columns follows the available panel width, and selecting a tile
opens the corresponding figure in the full-size detail view.

Interactive pyqtgraph panels appear as snapshot thumbnails rather than live
widgets. This keeps grid resizing responsive while the original widget retains
its hover, selection, and restyling behavior on its own tab. Activating a live
thumbnail raises that widget. Saved runs and interactive panels use the same
collapsible-section and workspace-state mechanisms.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..hidpi import follow_device_ratio, logical_size, scaled_for

#: Below this, a cell is too small to read anything in.
MIN_CELL_PX = 220
#: Above this, one figure eats the panel and the grid stops being a grid.
MAX_CELL_PX = 520
#: ONE SLOT PER FIGURE. Always.
#:
#: A grid whose cells are different sizes is not a grid, and the aspect ratio
#: is already preserved inside the cell:
#: a wide figure simply sits shorter in its slot, which is what a small
#: multiple should do.
CELL_SPAN = 1


def _letter_for(position: int) -> str:
    """A, B, ... Z, then AA. Publication lettering, not an index.

    Upper-case, no period -- the convention the published figures use and the
    one asked for by name.
    """
    letters = ""
    position += 1
    while position:
        position, remainder = divmod(position - 1, 26)
        letters = chr(ord("A") + remainder) + letters
    return letters


#: The width a single cell aims for. A regression run now produces eleven
#: panels or more, and at 320 a 740 px panel fits TWO of them -- six rows to
#: scroll through for one run. 250 fits three or four, which is the density
#: the published figures use and what makes a grid readable as one figure
#: rather than a list.
TARGET_CELL_PX = 230


def cells_across(panel_width: int, target: int = TARGET_CELL_PX) -> int:
    """How many cells fit across ``panel_width``.

    Widening the window should show MORE figures, not bigger ones -- the
    opposite of what a stretch-to-fit view does.
    """
    if panel_width <= 0:
        return 1
    return max(1, min(6, panel_width // max(target, MIN_CELL_PX)))


def cell_span(aspect: float) -> int:
    """Columns a figure occupies: one, whatever its shape.

    :param aspect: width / height. Accepted and deliberately ignored -- see
        :data:`CELL_SPAN`. Four plates take four slots.
    """
    return CELL_SPAN


#: How the run headings are drawn, minus the colour. One string, because the
#: chevron and the label are two widgets that have to read as one heading.
HEADING_STYLE = ("font-weight: 600; font-size: 11px; letter-spacing: 1px; "
                 "background: transparent;")

#: Fallback ink for a heading when the palette will not load -- a bare
#: process, a headless render. The theme's own accent, so the two agree.
_HEADING_FALLBACK = "#4A9EFF"


def _heading_style() -> str:
    """Return the section-heading stylesheet from the active palette.

    Resolving the accent at draw time keeps headings legible after a runtime
    theme change.
    """
    try:
        from ..theme import active_palette

        colour = active_palette()["accent"]
    except Exception:                                            # noqa: BLE001
        colour = _HEADING_FALLBACK
    return f"{HEADING_STYLE} color: {colour};"

#: How close to the top of the viewport a heading has to sit before the
#: gesture stops meaning "take me there". The console's tolerance, for the
#: console's reason: a scrollbar dragged by hand does not land on an exact
#: value. Shared as a name so the two cannot drift apart.
RAISED_TOLERANCE_PX = 4

#: What :meth:`FigureGridView.set_pinned` calls its one tile.
#:
#: A NAME, NOT A POSITION. While the live regression graph was the only live
#: tile, "the pinned tile" and "the first tile" were the same widget and the
#: distinction cost nothing. With a tile per pyqtgraph panel it stops being
#: free: "the first one" becomes whichever panel the caller happened to list
#: first, and ``pinned_activated`` -- which the regression screen wires
#: straight to "raise the interactive volcano" -- would open a Q-Q plot.
PINNED_KEY = "regression"

#: The heading the live tiles sit under, and the start index its section is
#: keyed at.
#:
#: -1 CANNOT COLLIDE WITH A RUN. A run's key is ``(label, start)`` where the
#: start is a position in the figure list, so it is never negative; the live
#: tiles are not a run and have no position in that list at all. Sharing the
#: section machinery rather than growing a second fold is the point -- one
#: collapsed set, one toggle, one scroll-to-top rule.
LIVE_SECTION_LABEL = "interactive"
LIVE_SECTION_START = -1


def live_tiles_from_panels(panels) -> list:
    """Photograph each pyqtgraph panel, ready for :meth:`set_live_tiles`.

    :param panels: ``[(key, title, widget)]`` -- the live panels, in the order
        they should appear. The widget only has to answer ``snapshot()``.
    :returns: ``[(key, pixmap, title)]`` for the ones that photographed.

    Panels whose ``snapshot()`` returns ``None`` are omitted rather than shown
    as empty, nonfunctional tiles. Snapshot errors also omit that panel for
    the current refresh so an optional preview cannot interrupt the screen.
    """
    tiles = []
    for entry in panels or ():
        key, title, widget = entry[0], entry[1], entry[2]
        if widget is None:
            continue
        try:
            pixmap = widget.snapshot()
        except Exception:                                        # noqa: BLE001
            continue
        if pixmap is None or pixmap.isNull():
            continue
        tiles.append((str(key), pixmap, str(title)))
    return tiles


class _SectionHeader(QFrame):
    """Provide a keyboard-accessible fold control for one figure section.

    The full bar toggles on click, Return, Enter, or Space and displays a
    disclosure chevron. ``section_key`` identifies the section as
    ``(label, start)`` so collapse state survives header reconstruction.
    """

    def __init__(self, label: str, key, parent=None, expanded: bool = True):
        """Build one collapsible section heading for the figure grid.

        :param label: the heading text.
        :param key: what identifies this section to the view, kept as
            ``section_key`` so expansion state survives a rebuild.
        :param parent: parent widget.
        :param expanded: whether the section starts open.

        The heading is a control: it takes a pointing hand and strong focus
        so it is reachable from the keyboard, and paints no background of
        its own so the grid shows through.
        """
        super().__init__(parent)
        self.setObjectName("FigureGridSectionHeader")
        self.section_key = key
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setStyleSheet("_SectionHeader { background: transparent; }")

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(6)
        self._chevron = QLabel("▾")
        self._chevron.setObjectName("FigureGridSectionChevron")
        # A run heading is generated text -- a timestamp, or a trial's name --
        # in whatever language was active when the run started. A later
        # whole-window language switch must not try to reinterpret it.
        self._chevron.setProperty("i18nSkipText", True)
        self._chevron.setStyleSheet(_heading_style())
        row.addWidget(self._chevron)
        self._label = QLabel(label)
        self._label.setObjectName("FigureGridSectionLabel")
        self._label.setProperty("i18nSkipText", True)
        self._label.setStyleSheet(_heading_style())
        row.addWidget(self._label)
        row.addStretch(1)
        self.set_expanded(expanded)

    def text(self) -> str:
        """The heading text, without the chevron."""
        return self._label.text()

    def is_expanded(self) -> bool:
        """Whether this run's figures are showing."""
        return self._expanded

    def set_expanded(self, expanded: bool) -> None:
        """Record the state and turn the chevron to match."""
        self._expanded = bool(expanded)
        self._chevron.setText("▾" if self._expanded else "▸")
        self.setToolTip(
            "Fold this run's figures away." if self._expanded
            else "Show this run's figures again.")

    def _view(self):
        """The owning :class:`FigureGridView`, or ``None``.

        Found by walking up rather than stored, exactly as the console's
        topic bar does it: the header is built inside a layout pass and the
        view is whichever ancestor knows how to toggle a section.
        """
        node = self.parentWidget()
        while node is not None and not hasattr(node, "toggle_section"):
            node = node.parentWidget()
        return node

    def _activate(self) -> None:
        """Fold or unfold this section, if the view is still there."""
        view = self._view()
        if view is not None:
            view.toggle_section(self)

    def mouseReleaseEvent(self, event):         # noqa: N802 - Qt naming
        # Release rather than press, so dragging off the bar cancels -- what
        # every other clickable in the app does.
        if (event.button() == Qt.LeftButton
                and self.rect().contains(event.position().toPoint())):
            self._activate()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):             # noqa: N802 - Qt naming
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self._activate()
            return
        super().keyPressEvent(event)


class _FigureCell(QFrame):
    """One figure, drawn at its own aspect ratio inside its cell.

    :ivar index: the position in the pixmap list handed to
        :meth:`FigureGridView.set_figures`, which is what
        ``figure_activated`` carries and what ``FigureQueue.show_index``
        consumes. ``-1`` on a live tile, which has no position in that list.
    :ivar live_key: which pyqtgraph panel this tile photographs, or ``""``
        for a figure tile. THE KEY IS THE IDENTITY OF A LIVE TILE, because
        its index cannot be: every live tile carries ``-1`` on purpose so
        that none of them can ever be mistaken for a figure.
    """

    clicked = Signal(int)
    #: index, global position -- the tile was right-clicked.
    menu_requested = Signal(int, object)

    def __init__(self, index: int, pixmap: QPixmap, title: str = "",
                 parent=None, letter: str = "", live_key: str = ""):
        """Build one tile.

        :param index: position in the pixmap list given to
            :meth:`FigureGridView.set_figures`, carried by
            ``figure_activated``. Pass ``-1`` for a live tile, which has no
            position in that list.
        :param pixmap: the figure to draw, at its own aspect ratio.
        :param title: the caption shown with the tile.
        :param parent: parent widget.
        :param letter: the panel letter, when the tile is part of a lettered
            figure.
        :param live_key: which pyqtgraph panel this tile photographs, or
            ``""`` for a figure tile. THE IDENTITY OF A LIVE TILE, because
            its index cannot be -- every live tile carries ``-1`` so none can
            be mistaken for a figure.
        """
        super().__init__(parent)
        self.index = index
        self.letter = letter
        self.live_key = live_key
        self._pixmap = pixmap
        #: The last width :meth:`fit_to` was given, so the cell can be drawn
        #: again at a new pixel density without the grid re-measuring.
        self._fit_width = 0
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        # "all gigures should be editable by right clicking" -- a tile is a
        # figure, so the gesture has to work here too and not only on the one
        # figure that happens to be open.
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._request_menu)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # A TILE DOES NOT PAINT ITS OWN GROUND. Reported as "on the grid (all
        # figures) the graphs still have a black background": the figures are
        # transparent and the frame behind them was not, so every tile was a
        # slab. The frame stays for its border; only its fill goes.
        self.setAutoFillBackground(False)
        self.setStyleSheet("_FigureCell { background: transparent; }")
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        if letter:
            # UPPER-CASE PANEL LETTER, top left, bold -- asked for by name:
            # "i asked you to make the all figures pannel publication style
            # (with each panel having an uppercase letter) and be on a grid".
            tag = QLabel(letter.upper())
            tag.setStyleSheet(
                "font-weight: 700; font-size: 15px; background: transparent;")
            tag.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            layout.addWidget(tag)

        self._image = QLabel()
        self._image.setAttribute(Qt.WA_TranslucentBackground, True)
        self._image.setStyleSheet("background: transparent;")
        self._image.setAlignment(Qt.AlignCenter)
        # NOT setScaledContents: that is exactly the stretch this replaces.
        # The pixmap is scaled with KeepAspectRatio when the cell is sized.
        self._image.setMinimumHeight(80)
        # A grid dragged onto a denser screen keeps its cell widths, so no
        # relayout arrives to refit the figures -- this is what does.
        follow_device_ratio(self._image, self._refit)
        layout.addWidget(self._image, 1)

        if title:
            caption = QLabel(title)
            caption.setAlignment(Qt.AlignCenter)
            caption.setWordWrap(True)
            caption.setStyleSheet("color: palette(mid); font-size: 10px;")
            layout.addWidget(caption)

    def aspect(self) -> float:
        if self._pixmap.isNull() or not self._pixmap.height():
            return 1.0
        return self._pixmap.width() / self._pixmap.height()

    def fit_to(self, width: int) -> None:
        """Scale the figure into ``width`` LOGICAL px, keeping its shape."""
        if self._pixmap.isNull() or width <= 0:
            return
        self._fit_width = int(width)
        scaled = scaled_for(
            self._pixmap, self._image,
            QSize(width, int(width / max(self.aspect(), 0.05))))
        self._image.setPixmap(scaled)
        # The height the cell reserves is what the picture OCCUPIES, not how
        # many pixels it was drawn with -- those differ by the ratio, and a
        # cell sized in device pixels is twice as tall as its picture.
        self._image.setFixedHeight(logical_size(scaled).height())

    def _refit(self) -> None:
        """Draw the figure again at the width it was last fitted to."""
        if self._fit_width > 0:
            self.fit_to(self._fit_width)

    def _request_menu(self, point) -> None:
        """Ask for this cell's context menu, in GLOBAL coordinates.

        The menu is placed by the view, which does not share this cell's
        coordinate space -- handing it a local point puts the menu in the wrong
        place on every cell but the first.
        """
        self.menu_requested.emit(self.index, self.mapToGlobal(point))

    def mousePressEvent(self, event):  # noqa: N802 - Qt naming
        # A right-click opens the menu; it must not ALSO open the figure, or
        # every attempt to restyle a tile navigates away from the grid first.
        if event.button() == Qt.RightButton:
            super().mousePressEvent(event)
            return
        self.clicked.emit(self.index)
        super().mousePressEvent(event)


class FigureGridView(QScrollArea):
    """Every figure at once, scrollable, each at its own aspect ratio.

    :ivar figure_activated: emitted with a figure's index when its cell is
        clicked, so the caller can open it full size.
    :ivar figure_menu_requested: emitted with (index, global position) when a
        cell is right-clicked. The grid holds pictures, not figures, so the
        menu itself is the caller's to build -- it is the one that still has
        the matplotlib object.

    :param parent: parent widget.
    """

    figure_activated = Signal(int)
    figure_menu_requested = Signal(int, object)
    #: Emitted when the PINNED tile is pressed. Separate from
    #: ``figure_activated`` because the pinned tile is not one of the run's
    #: figures and has no index among them -- sharing the signal would mean a
    #: sentinel index, and a sentinel index is a wrong figure waiting to be
    #: opened by whoever forgets to check for it.
    pinned_activated = Signal()
    #: Emitted with a global position when the PINNED tile is right-clicked.
    #: Separate from ``figure_menu_requested`` for the reason above: the queue
    #: builds that menu from a matplotlib figure at that index, and the pinned
    #: tile is not at any index and is not a matplotlib figure.
    pinned_menu_requested = Signal(object)
    #: Emitted with a live tile's KEY when it is pressed. The general form of
    #: ``pinned_activated``: the volcano is no longer the only interactive
    #: graph on the grid, and a caller has to know WHICH panel to raise. A key
    #: rather than a position, because the set of panels a run can support
    #: varies -- a fit with no model has no residual plot -- so position 3 is
    #: a different graph on two different runs.
    live_tile_activated = Signal(str)
    #: Emitted with (key, global position) when a live tile is right-clicked.
    #: Each pyqtgraph panel builds its own restyle menu, so the caller needs
    #: the key to ask the right one.
    live_tile_menu_requested = Signal(str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self._body = QWidget()
        self._grid = QGridLayout(self._body)
        self._grid.setContentsMargins(6, 6, 6, 6)
        self._grid.setSpacing(8)
        self._grid.setAlignment(Qt.AlignTop)
        self.setWidget(self._body)

        self._cells: list[_FigureCell] = []
        #: The pyqtgraph tiles, in the order the caller listed them. NEVER in
        #: ``_cells``: see :meth:`set_live_tiles` for why that separation is
        #: the whole of the index mapping.
        self._live: list[_FigureCell] = []
        self._sections: list = []
        #: The headings currently in the layout. Owned explicitly because
        #: :meth:`_relayout` has to destroy the previous ones -- see there.
        self._headers: list = []
        #: ``{(label, start)}`` the user has folded away. THE STATE LIVES
        #: HERE, not on the header widget, and that is the whole design:
        #: every relayout builds new headers, and a run that re-expanded
        #: everything each time it finished would be unusable during a sweep
        #: of sixty trials. A section not in this set is open, so a run that
        #: has never been seen before arrives open -- the newest one is the
        #: one that matters.
        self._collapsed: set = set()
        self._target = TARGET_CELL_PX

    @property
    def _pinned(self) -> Optional[_FigureCell]:
        """The live REGRESSION tile, or ``None``.

        Derived rather than stored, and derived by KEY rather than by
        position. There used to be exactly one live tile and this was a plain
        attribute; now there is a tile per pyqtgraph panel and "the pinned
        one" has to keep meaning the regression graph however many panels sit
        beside it, or the screen's "open the interactive volcano" wiring opens
        whichever tile the caller listed first.
        """
        for cell in self._live:
            if cell.live_key == PINNED_KEY:
                return cell
        return None

    def live_tile_keys(self) -> list:
        """Return live-panel keys in display order.

        Panels without a usable snapshot are omitted, matching
        :meth:`set_live_tiles`.
        """
        return [cell.live_key for cell in self._live]

    def set_live_tiles(self, tiles) -> int:
        """Replace the foldable section of live-panel snapshots.

        Parameters
        ----------
        tiles : iterable
            ``(key, pixmap)`` or ``(key, pixmap, title)`` entries. Activating
            a tile emits its key through :attr:`live_tile_activated`.

        Returns
        -------
        int
            Number of snapshots retained on the grid.

        Notes
        -----
        Entries without a pixmap are omitted. Existing tile widgets are
        destroyed before the complete set is rebuilt so refreshes cannot
        stack stale snapshots or retain panels no longer available.
        """
        previous = self._live
        rebuilt: list = []
        for entry in tiles or ():
            key = str(entry[0])
            pixmap = entry[1]
            title = str(entry[2]) if len(entry) > 2 else ""
            if pixmap is None or pixmap.isNull():
                continue
            cell = _FigureCell(-1, pixmap, title, self._body, live_key=key)
            cell.clicked.connect(
                lambda _index, _key=key: self._live_activated(_key))
            cell.menu_requested.connect(
                lambda _index, position, _key=key:
                    self._live_menu(_key, position))
            rebuilt.append(cell)
        self._live = rebuilt
        # BEFORE the relayout, not after: `_discard` is what takes a tile off
        # the body, and a tile still parented to the body when the layout runs
        # paints itself at its old geometry for the rest of the event-loop
        # turn. That ordering is the fix, not the discarding.
        for cell in previous:
            self._discard(cell)
        self._relayout()
        return len(self._live)

    def _live_activated(self, key: str) -> None:
        """A live tile was pressed: say which one.

        ``pinned_activated`` is emitted as well for the regression tile, and
        that is not redundancy for its own sake -- it is the one signal the
        regression screen is already wired to, and a screen that has not yet
        learned about the other panels must go on working.
        """
        self.live_tile_activated.emit(key)
        if key == PINNED_KEY:
            self.pinned_activated.emit()

    def _live_menu(self, key: str, position) -> None:
        """A live tile was right-clicked: say which one, and where."""
        self.live_tile_menu_requested.emit(key, position)
        if key == PINNED_KEY:
            self.pinned_menu_requested.emit(position)

    def set_pinned(self, pixmap, title: str = "") -> bool:
        """A tile that is always first and is not one of the run's figures.

        The tile is a snapshot of the interactive regression graph. Activating
        it opens the live widget. It occupies a separate slot from ``_cells``
        so persisted figure indices remain aligned with
        :meth:`FigureQueue.show_index`; activation uses
        :attr:`pinned_activated` instead of a sentinel figure index.

        Replacing the pinned tile destroys the previous cell before relayout,
        preventing transparent snapshots from accumulating at the same grid
        position.

        This method updates only the regression tile; other live-panel tiles
        remain in place. Use :meth:`set_live_tiles` to replace the full live
        section.

        :returns: True when a tile was pinned. A null or missing pixmap
            removes it.
        """
        previous = self._pinned
        others = [cell for cell in self._live if cell is not previous]
        if pixmap is None or pixmap.isNull():
            self._live = others
            self._discard(previous)
            self._relayout()
            return False
        cell = _FigureCell(-1, pixmap, title, self._body,
                           live_key=PINNED_KEY)
        cell.clicked.connect(
            lambda _index: self._live_activated(PINNED_KEY))
        # "all gigures should be editable by right clicking" -- and this one
        # is the only tile on the grid that is a real, live figure, so a
        # right-click that did nothing here would be the gesture failing on
        # the one tile where it can do the most.
        cell.menu_requested.connect(
            lambda _index, position: self._live_menu(PINNED_KEY, position))
        # FIRST, whatever else is on the section. "always first" is what the
        # name promises and what the caller relies on to find it.
        self._live = [cell] + others
        self._discard(previous)
        self._relayout()
        return True

    @staticmethod
    def _discard(widget) -> None:
        """Take a tile off the body for good.

        UNPARENTING IS THE POINT, not `deleteLater` -- deletion is deferred to
        the next event-loop turn, and a widget that is still a child of the
        body until then goes on painting itself at its old geometry. Anything
        this view is finished with has to leave the body in the same call that
        finishes with it, or it is on screen for as long as nothing turns the
        loop.
        """
        if widget is None:
            return
        try:
            widget.setParent(None)
            widget.deleteLater()
        except RuntimeError:
            # Already torn down by Qt -- the screen closed under us.
            pass

    def set_target_cell_width(self, pixels: int) -> None:
        """How wide a single-width cell should be, before layout."""
        self._target = max(MIN_CELL_PX, min(int(pixels), MAX_CELL_PX))
        self._relayout()

    def clear(self) -> None:
        doomed = []
        live = set(map(id, self._live))
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            # The live tiles survive a clear: they are not the figures being
            # replaced, and a run that streams new ones must not make the
            # interactive graphs disappear. Compared by identity through a set
            # of ids rather than `in self._live` -- `in` on a list of QWidgets
            # goes through `__eq__`, which Qt does not define for widgets, so
            # it degrades to identity anyway but at O(n) per tile on a grid
            # that can hold a few hundred.
            if widget is not None and id(widget) not in live:
                doomed.append(widget)
        # THE LAYOUT IS NOT THE WHOLE GRID. A cell belonging to a FOLDED run is
        # deliberately left out of the layout by `_relayout` (so the next run
        # flows up under the folded heading instead of into a hole), which
        # means walking the layout alone never reaches it -- it stays a child
        # of the body while `_cells` is emptied out from under it, and the only
        # reference to it is gone. Nothing on screen, but it is still there,
        # and a sweep that folds its runs away leaks one per figure.
        seen = set(map(id, doomed))
        for cell in self._cells:
            if id(cell) not in live and id(cell) not in seen:
                doomed.append(cell)
        for widget in doomed:
            self._discard(widget)
        self._cells = []
        # The headings went out with the rest of the layout, so the list must
        # go too -- otherwise _relayout reaches through a wrapper whose C++
        # object has already been torn down and raises RuntimeError. The
        # COLLAPSED SET deliberately survives: clearing is how the grid is
        # rebuilt after every run, and a fold that came undone on each rebuild
        # is the unusable sweep this exists to prevent.
        self._headers = []

    def set_figures(self, pixmaps, titles=None, sections=None) -> int:
        """Replace the grid contents and return the number of figures added.

        Parameters
        ----------
        pixmaps : iterable
            Figure images to display.
        titles : iterable, optional
            Captions corresponding to ``pixmaps``.
        sections : iterable, optional
            ``(label, start, count)`` entries describing runs. Panel lettering
            restarts within each section.
        """
        self.clear()
        titles = list(titles or [])
        self._sections = list(sections or [])
        starts = {start: label for label, start, _count in self._sections}
        letter_at = 0
        for index, pixmap in enumerate(pixmaps):
            if index in starts:
                letter_at = 0
            if pixmap is None or pixmap.isNull():
                continue
            title = titles[index] if index < len(titles) else ""
            cell = _FigureCell(index, pixmap, title, self._body,
                               letter=_letter_for(letter_at))
            letter_at += 1
            cell.clicked.connect(self.figure_activated)
            cell.menu_requested.connect(self.figure_menu_requested)
            self._cells.append(cell)
        self._relayout()
        return len(self._cells)

    # ------------------------------------------------------------- sections

    @staticmethod
    def _section_key(label, start):
        """What identifies a run's section across relayouts and new runs.

        The label ALONE is not enough -- two runs a second apart can carry the
        same timestamp -- and the start index alone is not either, because a
        cleared queue starts a different run at 0. Together they are stable
        for as long as the section exists: figures are appended, so an
        existing run's start never moves.
        """
        return (str(label), int(start))

    # -- instruction 180: what the grid contributes to a saved run ----------

    def workspace_state(self) -> dict:
        """How the grid is ARRANGED, not what is in it.

        The figures themselves are files in the run's own results folder,
        recorded by the sections that name that folder; copying seventeen
        PNGs in here would be a second copy of something the run already
        has. What dies with the process is the arrangement -- the tile size
        the user settled on and which sections they folded away -- and a
        sweep of sixty trials is unusable if that resets.
        """
        return {
            "cell_width": int(self._target),
            "collapsed": [list(key) for key in sorted(self._collapsed, key=str)],
        }

    def apply_workspace_state(self, state) -> bool:
        """Put the arrangement back. Returns whether anything applied."""
        if not isinstance(state, dict):
            return False
        applied = False
        needs_relayout = False
        collapsed = state.get("collapsed")
        if isinstance(collapsed, list):
            self._collapsed = {tuple(key) if isinstance(key, (list, tuple))
                               else key for key in collapsed}
            applied = True
            needs_relayout = True
        width = state.get("cell_width")
        if width:
            try:
                # Through the setter: it clamps and relayouts, and a raw
                # `_target` would leave the grid drawn at the old width until
                # something else happened to trigger a relayout.
                self.set_target_cell_width(int(width))
                applied = True
                needs_relayout = False
            except (TypeError, ValueError):
                pass
        if needs_relayout:
            self._relayout()
        return applied

    def is_section_collapsed(self, label, start) -> bool:
        """Whether this run's figures are folded away."""
        return self._section_key(label, start) in self._collapsed

    def set_section_collapsed(self, label, start,
                               collapsed: bool = True) -> None:
        """Fold a run's figures away, or bring them back."""
        key = self._section_key(label, start)
        if collapsed:
            self._collapsed.add(key)
        else:
            self._collapsed.discard(key)
        self._relayout()

    def is_live_section_collapsed(self) -> bool:
        """Whether the pyqtgraph tiles are folded away.

        Named rather than left to :meth:`is_section_collapsed` with two magic
        arguments: the live section's key is a constant of this module and a
        caller repeating ``(LIVE_SECTION_LABEL, LIVE_SECTION_START)`` is a
        caller who can get one of them wrong and silently ask about a section
        that does not exist.
        """
        return self.is_section_collapsed(LIVE_SECTION_LABEL,
                                         LIVE_SECTION_START)

    def _hidden_indices(self) -> frozenset:
        """Figure indices belonging to a folded run.

        Only runs that actually GET a heading can be folded, so a key left
        over from an earlier grid cannot silently hide figures with no
        control on screen to bring them back. Every section gets one now --
        see `_relayout` -- so the guard is just "there are sections".
        """
        if not self._sections or not self._collapsed:
            return frozenset()
        hidden = set()
        for label, start, count in self._sections:
            if self._section_key(label, start) in self._collapsed:
                hidden.update(range(int(start), int(start) + int(count)))
        return frozenset(hidden)

    def _is_raised(self, header) -> bool:
        """Whether ``header`` is already sitting at the top of the viewport."""
        try:
            top = header.mapTo(self._body, header.rect().topLeft()).y()
            bar = self.verticalScrollBar()
        except RuntimeError:
            return False        # header rebuilt between click and query
        return abs(bar.value() - min(top, bar.maximum())) <= RAISED_TOLERANCE_PX

    def _scroll_section_to_top(self, key) -> bool:
        """Put the (rebuilt) header for ``key`` at the top of the viewport.

        Runs one event-loop turn after the toggle, so everything it touches
        may have been torn down in between -- the grid is rebuilt whenever a
        figure lands, and the screen can be closed on the same turn.
        """
        for header in self._headers:
            if getattr(header, "section_key", None) != key:
                continue
            try:
                top = header.mapTo(self._body, header.rect().topLeft()).y()
                bar = self.verticalScrollBar()
                bar.setValue(min(top, bar.maximum()))
            except RuntimeError:
                return False
            return True
        return False

    def toggle_section(self, header) -> bool:
        """Reach the run first; fold it away second. Returns the new state.

        The console provides the interaction model. A heading that is not
        already at the top of the
        viewport is a request to GO THERE, whatever its state -- a first click
        that hid the very section the user was reaching for spends the gesture
        on the opposite of what it looked like. Only a heading already at the
        top has nowhere left to navigate to, and there folding is the one
        thing the gesture can still mean, on a second click exactly where the
        user's hand already is.
        """
        key = getattr(header, "section_key", None)
        if key is None:
            return True
        if key not in self._collapsed and self._is_raised(header):
            self._collapsed.add(key)
            self._relayout()
            return False
        self._collapsed.discard(key)
        self._relayout()
        # After layout, not during: the geometry this scroll needs does not
        # exist until the cells just shown have been placed.
        QTimer.singleShot(0, lambda: self._scroll_section_to_top(key))
        return True

    def _relayout(self) -> None:
        """Place the cells, giving wide figures a double-width cell."""
        # THE PREVIOUS HEADINGS ARE DESTROYED, NOT MERELY UNPARENTED FROM THE
        # LAYOUT. `takeAt` removes the layout item and leaves the widget a
        # visible child of the body at its old geometry, so every relayout --
        # and a window resize is a relayout -- used to leave another copy of
        # every run heading painted on the grid. Measured before the fix:
        # three relayouts of a two-run grid left six headings. The pinned tile
        # had the same bug for the same reason -- see :meth:`set_pinned`, which
        # now shares this one's cleanup rather than re-deriving it a third
        # time.
        for header in self._headers:
            self._discard(header)
        self._headers = []
        for index in reversed(range(self._grid.count())):
            self._grid.takeAt(index)

        columns = cells_across(self.viewport().width(), self._target)
        available = max(self.viewport().width() - 24, MIN_CELL_PX)
        unit = max(available // columns, MIN_CELL_PX // 2)

        # A HEADING PER RUN, INCLUDING THE FIRST AND ONLY ONE.
        #
        # It used to appear only from the second run onwards, on the argument
        # that the lettering restarting is what needs explaining and one run
        # never restarts. That argument was about the LABEL and this control
        # is also the fold: with one run there was no header, so there was
        # nothing to click, and the maintainer reported the figures as "still
        # not colapsable into runs" while the folding worked perfectly from
        # the second run on.
        #
        # A heading over a single run costs one row and answers "which run is
        # this" -- which the grid could not previously say at all.
        heading_at = {}
        for label, start, _count in self._sections:
            heading_at[start] = label
        hidden = self._hidden_indices()

        row, column = self._lay_out_live_section(columns, unit)
        for cell in self._cells:
            index = getattr(cell, "index", -1)
            if index in heading_at:
                if column:
                    row, column = row + 1, 0
                label = heading_at.pop(index)
                key = self._section_key(label, index)
                header = _SectionHeader(label, key, self._body,
                                        expanded=key not in self._collapsed)
                self._grid.addWidget(header, row, 0, 1, max(columns, 1))
                header.setVisible(True)
                self._headers.append(header)
                row, column = row + 1, 0
            if index in hidden:
                # Left out of the layout AND hidden. Out of the layout so the
                # next run flows up under the folded heading instead of into
                # a hole; hidden because a widget removed from a layout keeps
                # painting itself where it last was.
                cell.setVisible(False)
                continue
            cell.setVisible(True)
            span = min(cell_span(cell.aspect()), columns)
            # A wide figure that will not fit in what is left of this row
            # starts the next one, rather than being squeezed.
            if column + span > columns:
                row, column = row + 1, 0
            self._grid.addWidget(cell, row, column, 1, span)
            cell.fit_to(unit * span - 16)
            column += span
            if column >= columns:
                row, column = row + 1, 0

        for index in range(columns):
            self._grid.setColumnStretch(index, 1)

    def _lay_out_live_section(self, columns: int, unit: int) -> tuple:
        """Place the pyqtgraph tiles and their heading. Returns (row, column).

        FIRST ON THE GRID, and under a heading that folds like a run's.

        The heading is not decoration. A full panel set is eight tiles, which
        on a 740 px panel is three rows of the grid before the run's own
        figures begin -- and 125 C's argument for folding a run ("a sweep of
        sixty trials that re-expanded everything would be unusable") applies
        exactly: a user comparing this run's plate heatmaps wants the
        interactive block out of the way, and there was previously no control
        that could put it there. It is the SAME control as a run's, through
        the same `_collapsed` set and the same `toggle_section`, rather than a
        second opinion about what a foldable section looks like.

        No tiles, no heading: an empty section with a chevron is a fold
        control for nothing, and it would appear on every module screen in the
        application, none of which has a live plot at all.

        The row is broken on the way out so a run's heading starts a fresh
        one -- a heading sharing a row with the last live tile reads as a
        caption for it.
        """
        if not self._live:
            return 0, 0
        collapsed = self.is_live_section_collapsed()
        key = self._section_key(LIVE_SECTION_LABEL, LIVE_SECTION_START)
        header = _SectionHeader(LIVE_SECTION_LABEL, key, self._body,
                                expanded=not collapsed)
        self._grid.addWidget(header, 0, 0, 1, max(columns, 1))
        header.setVisible(True)
        self._headers.append(header)

        row, column = 1, 0
        for cell in self._live:
            if collapsed:
                # Out of the layout AND hidden, for the reason the folded
                # figures are: a widget merely removed from a layout goes on
                # painting itself where it last was.
                cell.setVisible(False)
                continue
            cell.setVisible(True)
            self._grid.addWidget(cell, row, column, 1, CELL_SPAN)
            cell.fit_to(unit * CELL_SPAN - 16)
            column += CELL_SPAN
            if column >= columns:
                row, column = row + 1, 0
        return (row + 1, 0) if column else (row, 0)

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._relayout()
