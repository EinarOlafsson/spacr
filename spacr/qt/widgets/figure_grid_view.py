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

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
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
        self._target = TARGET_CELL_PX

    def set_pinned(self, pixmap, title: str = "") -> bool:
        """A tile that is always first and is not one of the run's figures.

        The regression graph is a LIVE widget, not a picture the pipeline
        saved, and it is the one the maintainer asked to be interactive. Left
        to the ordinary path the grid would show the pipeline's static copy of
        the same plot -- two volcanoes on screen, the big one dead -- so the
        live one takes the first tile and the press opens the real thing.
        """
        if pixmap is None or pixmap.isNull():
            self._pinned = None
            self._relayout()
            return False
        cell = _FigureCell(-1, pixmap, title, self._body)
        cell.clicked.connect(lambda _index: self.pinned_activated.emit())
        self._pinned = cell
        self._relayout()
        return True

    def set_target_cell_width(self, pixels: int) -> None:
        """How wide a single-width cell should be, before layout."""
        self._target = max(MIN_CELL_PX, min(int(pixels), MAX_CELL_PX))
        self._relayout()

    def clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            # The pinned tile survives a clear: it is not one of the figures
            # being replaced, and a run that streams new ones must not make
            # the interactive graph disappear.
            if widget is not None and widget is not self._pinned:
                widget.setParent(None)
                widget.deleteLater()
        self._cells = []

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

    def _relayout(self) -> None:
        """Place the cells, giving wide figures a double-width cell."""
        for index in reversed(range(self._grid.count())):
            self._grid.takeAt(index)

        columns = cells_across(self.viewport().width(), self._target)
        available = max(self.viewport().width() - 24, MIN_CELL_PX)
        unit = max(available // columns, MIN_CELL_PX // 2)

        # A HEADING PER RUN. The lettering restarting is only legible if the
        # reader can see where one run ends and the next begins -- otherwise
        # two panels called A look like a bug rather than two figures.
        heading_at = {}
        if len(self._sections) > 1:
            for label, start, _count in self._sections:
                heading_at[start] = label

        row = column = 0
        for cell in ([self._pinned] if self._pinned is not None else []) \
                + self._cells:
            index = getattr(cell, "index", -1)
            if index in heading_at:
                if column:
                    row, column = row + 1, 0
                heading = QLabel(heading_at.pop(index))
                heading.setStyleSheet(
                    "font-weight: 600; font-size: 11px; letter-spacing: 1px; "
                    "color: palette(mid); background: transparent;")
                self._grid.addWidget(heading, row, 0, 1, max(columns, 1))
                row, column = row + 1, 0
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
