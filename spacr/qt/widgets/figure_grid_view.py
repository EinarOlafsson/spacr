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
#: A figure wider than this many times its height is treated as a WIDE figure
#: and given a double-width cell. A plate is 24x16 wells (1.5); a volcano is
#: square. 1.35 separates them without needing to know which is which.
WIDE_ASPECT = 1.35


def cells_across(panel_width: int, target: int = 320) -> int:
    """How many cells fit across ``panel_width``.

    Widening the window should show MORE figures, not bigger ones -- the
    opposite of what a stretch-to-fit view does.
    """
    if panel_width <= 0:
        return 1
    return max(1, min(6, panel_width // max(target, MIN_CELL_PX)))


def cell_span(aspect: float) -> int:
    """Columns a figure of this aspect ratio should occupy.

    :param aspect: width / height.
    """
    if aspect >= WIDE_ASPECT:
        return 2
    return 1


class _FigureCell(QFrame):
    """One figure, drawn at its own aspect ratio inside its cell."""

    clicked = Signal(int)

    def __init__(self, index: int, pixmap: QPixmap, title: str = "",
                 parent=None):
        super().__init__(parent)
        self.index = index
        self._pixmap = pixmap
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._image = QLabel()
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

    def mousePressEvent(self, event):  # noqa: N802 - Qt naming
        self.clicked.emit(self.index)
        super().mousePressEvent(event)


class FigureGridView(QScrollArea):
    """Every figure at once, scrollable, each at its own aspect ratio.

    :ivar figure_activated: emitted with a figure's index when its cell is
        clicked, so the caller can open it full size.
    """

    figure_activated = Signal(int)

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
        self._target = 320

    def set_target_cell_width(self, pixels: int) -> None:
        """How wide a single-width cell should be, before layout."""
        self._target = max(MIN_CELL_PX, min(int(pixels), MAX_CELL_PX))
        self._relayout()

    def clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._cells = []

    def set_figures(self, pixmaps, titles=None) -> int:
        """Show these figures. Returns how many were added."""
        self.clear()
        titles = list(titles or [])
        for index, pixmap in enumerate(pixmaps):
            if pixmap is None or pixmap.isNull():
                continue
            title = titles[index] if index < len(titles) else ""
            cell = _FigureCell(index, pixmap, title, self._body)
            cell.clicked.connect(self.figure_activated)
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

        row = column = 0
        for cell in self._cells:
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
