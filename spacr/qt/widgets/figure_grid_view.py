"""A run's figures on a scrollable grid, each at its own aspect ratio.

WHY THIS EXISTS

The figures panel shows ONE figure at a time, fitted into whatever shape the
panel happens to be. A regression run produces seventeen; seeing the fourth
means clicking to it, and every one is stretched to a container that has
nothing to do with its own proportions.

That is not merely untidy. A plate heatmap distorted into a square is no
longer a heatmap of a plate -- the wells stop being square, and positional
artefacts, the entire reason to look at one, become impossible to see. And a
run's figures are meant to be read together: the fraction histogram explains
the volcano, and one-at-a-time navigation hides the relationship.

So: a grid that scrolls, cells sized from the panel width, and every figure
drawn at the aspect ratio it was created with. Wide figures take a wide cell
and fewer per row; square ones tile. Clicking a cell opens that figure full
size, and the existing one-at-a-time view is that detail view.
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy,
    QVBoxLayout, QWidget,
)

#: Below this, a cell is too small to read anything in.
MIN_CELL_PX = 220
#: Above this, one figure eats the panel and the grid stops being a grid.
MAX_CELL_PX = 520
#: ONE SLOT PER FIGURE. Always.
#:
#: This used to give a wide figure a DOUBLE-width cell, so four plate
#: heatmaps took eight slots and wrapped onto two rows -- reported as "the
#: plate heat maps are too wide so now they take 2 slots ... when they should
#: take 1 slot per plate so in my case 4 slots".
#:
#: A grid whose cells are different sizes is not a grid, and the aspect ratio
#: is already preserved INSIDE the cell (that is what instruction 117 fixed):
#: a wide figure simply sits shorter in its slot, which is what a small
#: multiple should do. Kept as a name rather than deleted so the old rule
#: cannot quietly come back as a literal.
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
    """The heading stylesheet, coloured from the live palette.

    BLUE, NOT GREY. It was `color: palette(mid)`, which is Qt's mid role: a
    mid-grey that is legible on the light theme and, in the maintainer's
    words, "barely visable" on dark -- where every spaCR theme but one lives.
    A heading that cannot be read is a fold control that cannot be found,
    which is the second time this heading has been invisible for a different
    reason.

    Read at DRAW TIME rather than baked into a module constant, so the
    heading follows a theme switch. `palette(mid)` at least had that
    property and a hex literal would lose it; this keeps it. Same pattern as
    the clear-figures control.
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


class _SectionHeader(QFrame):
    """A run's heading, and the control that folds that run's figures away.

    THE CONSOLE'S GESTURE, because the maintainer named the console as the
    model: "each set should be in its own section that can be minimized like
    in the console". So this is `_TopicBar` in every respect that a user can
    see -- a disclosure chevron on the left of the heading text, a pointing
    hand over the whole bar, strong focus so the keyboard reaches it, and
    Return / Enter / Space doing what a click does. A control only a mouse can
    reach is one some users cannot reach at all.

    Only the CHEVRON and the click are new. The heading itself, its wording
    and its styling are what 124 B already drew; making it a control must not
    make it a different heading.

    :ivar section_key: ``(label, start)`` -- which run this heads. The header
        WIDGET is rebuilt on every relayout (a resize rebuilds the grid), so
        the collapsed state cannot live on it; the key is what the view
        remembers instead.
    """

    def __init__(self, label: str, key, parent=None, expanded: bool = True):
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
    """One figure, drawn at its own aspect ratio inside its cell."""

    clicked = Signal(int)
    #: index, global position -- the tile was right-clicked.
    menu_requested = Signal(int, object)

    def __init__(self, index: int, pixmap: QPixmap, title: str = "",
                 parent=None, letter: str = ""):
        super().__init__(parent)
        self.index = index
        self.letter = letter
        self._pixmap = pixmap
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
        """Scale the figure into ``width``, keeping its own proportions."""
        if self._pixmap.isNull() or width <= 0:
            return
        scaled = self._pixmap.scaled(
            QSize(width, int(width / max(self.aspect(), 0.05))),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._image.setPixmap(scaled)
        self._image.setFixedHeight(scaled.height())

    def _request_menu(self, point) -> None:
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
        self._pinned: Optional[_FigureCell] = None
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

    def set_pinned(self, pixmap, title: str = "") -> bool:
        """A tile that is always first and is not one of the run's figures.

        The regression graph is a LIVE widget, not a picture the pipeline
        saved, and it is the one the maintainer asked to be interactive:
        "the regression plot isnt shown in all figures (i want it also shown
        there)". Pressing this tile opens the real widget, not a picture of
        it.

        THE INDEX MAPPING SURVIVES THIS, and that is the whole reason the
        pinned tile is a separate slot rather than an extra entry in
        ``_cells``. ``_FigureCell.index`` is the position in the pixmap list
        the caller handed to :meth:`set_figures`, and ``figure_activated``
        forwards it straight to ``FigureQueue.show_index``; anything INSERTED
        into that list shifts every figure after it, so every tile would open
        its neighbour. This cell carries ``-1``, is never in ``_cells``, and
        emits :attr:`pinned_activated` instead -- a sentinel index down the
        shared signal would be the same bug with an extra step.

        THE TILE IT REPLACES IS DESTROYED, and that is the whole of the
        stacked-volcano bug reported on 2026-08-17: "the thumbnail iage of the
        volcano plot looks like several volcano plot itterations pasted on top
        of each other". This method used to rebind ``_pinned`` and relayout,
        and `_relayout`'s `takeAt` removes a widget from the LAYOUT while
        leaving it a visible child of the body at its old geometry -- exactly
        the failure already written down for the section headings a few lines
        below, re-derived here for the tiles. `clear()` could not collect the
        strays either: it walks the layout, and a stray is no longer in it.
        `_pin_regression_graph` runs on every grid refresh, so a screen that
        had done twelve runs was painting a dozen tiles at (6, 6) at once --
        and `FastPlot.snapshot` returns a TRANSPARENT pixmap, so every one of
        them showed through the ones in front. Measured on the real widget:
        five `set_pinned` calls left five visible cells at identical geometry
        and all five volcanoes painted at once; one after the fix.

        :returns: True when a tile was pinned. A null or missing pixmap
            REMOVES it, because an empty tile invites a click that opens an
            empty plot.
        """
        previous = self._pinned
        if pixmap is None or pixmap.isNull():
            self._pinned = None
            self._discard(previous)
            self._relayout()
            return False
        cell = _FigureCell(-1, pixmap, title, self._body)
        cell.clicked.connect(lambda _index: self.pinned_activated.emit())
        # "all gigures should be editable by right clicking" -- and this one
        # is the only tile on the grid that is a real, live figure, so a
        # right-click that did nothing here would be the gesture failing on
        # the one tile where it can do the most.
        cell.menu_requested.connect(
            lambda _index, position: self.pinned_menu_requested.emit(position))
        self._pinned = cell
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
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            # The pinned tile survives a clear: it is not one of the figures
            # being replaced, and a run that streams new ones must not make
            # the interactive graph disappear.
            if widget is not None and widget is not self._pinned:
                doomed.append(widget)
        # THE LAYOUT IS NOT THE WHOLE GRID. A cell belonging to a FOLDED run is
        # deliberately left out of the layout by `_relayout` (so the next run
        # flows up under the folded heading instead of into a hole), which
        # means walking the layout alone never reaches it -- it stays a child
        # of the body while `_cells` is emptied out from under it, and the only
        # reference to it is gone. Nothing on screen, but it is still there,
        # and a sweep that folds its runs away leaks one per figure.
        for cell in self._cells:
            if cell is not self._pinned and cell not in doomed:
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
        """Show these figures. Returns how many were added.

        :param sections: ``[(label, start, count)]`` -- one entry per run.
            LETTERING RESTARTS IN EACH, because a panel letter belongs to a
            figure and a figure is one run's worth of panels. Without this a
            second run continues at L, which says nothing to a reader and
            was reported as exactly that.
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

        THE CONSOLE'S RULE, verbatim, because the console is the model the
        maintainer named. A heading that is not already at the top of the
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

        row = column = 0
        for cell in ([self._pinned] if self._pinned is not None else []) \
                + self._cells:
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

    def resizeEvent(self, event):  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._relayout()
