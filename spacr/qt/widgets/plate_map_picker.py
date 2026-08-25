"""Select one or more wells from a plate-shaped grid.

Selections are parsed and serialized through :mod:`spacr.well_spec`, so rows,
columns, and individual wells use the same validated vocabulary in the GUI and
headless APIs. Rectangular selection is exposed through :meth:`select_region`
for mouse-drag handling and programmatic use.

ONE GRID, NOT THREE THINGS NEAR EACH OTHER. The column numbers, the row
letters and the wells all live in a single :class:`QGridLayout`, every item
centred in its cell, so a column number sits over its column and a row letter
beside its row at every window size. The grid takes the width and height its
square wells need and leaves the rest to a trailing empty row and column,
rather than dividing whatever the window happens to give it -- which is what
pulls the labels away from the wells they name.
"""
from __future__ import annotations

from typing import Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QGridLayout,
                               QHBoxLayout, QInputDialog, QLabel,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

from ...well_spec import (DEFAULT_LAYOUT, LAYOUTS, WellSpecError, parse,
                          row_label, shape, to_text, well_label)

#: The blue a chosen well is filled with, and the ground it sits on.
CHOSEN = "#4472C4"
EMPTY = "rgba(255, 255, 255, 0.06)"

#: The side of a well, in pixels. A well is a SQUARE at every window size --
#: a plate map is a picture of a physical object, and a well stretched into a
#: rectangle reads as a grid of something else. Pinning the side is what makes
#: the grid ask for the space it needs instead of dividing what it is handed.
WELL_SIDE = 22


class _Well(QPushButton):
    """One well. Checkable, so its state IS the selection."""

    def __init__(self, row: int, column: int, parent=None):
        super().__init__(parent)
        self.row, self.column = int(row), int(column)
        self.setCheckable(True)
        self.setFixedSize(WELL_SIDE, WELL_SIDE)
        self.setToolTip(well_label(row, column))
        self._paint()
        self.toggled.connect(lambda *_: self._paint())

    # ---------------------------------------------------------- the drag
    #
    # THE PRESSED BUTTON RECEIVES EVERY MOVE, because Qt grabs the mouse on
    # a press -- so a sibling's `enterEvent` never fires while a drag is in
    # progress, and the well under the pointer has to be found by asking the
    # grid rather than by waiting to be told.

    def _picker(self):
        """The :class:`PlateMapPicker` this well belongs to, or ``None``.

        FOUND BY ANCESTRY, NEVER BY COUNTING STEPS. A scroll area reparents
        the widget it is given into its own viewport, so a fixed two-parent
        walk lands on that viewport, every ``hasattr(picker, "begin_drag")``
        guard below reads False, and the drag gesture silently does nothing.
        """
        node = self.parent()
        while node is not None:
            if isinstance(node, PlateMapPicker):
                return node
            node = node.parent()
        return None

    def mousePressEvent(self, event):                # noqa: N802 - Qt
        picker = self._picker()
        if picker is not None and hasattr(picker, "begin_drag"):
            picker.begin_drag(self.row, self.column, event.modifiers())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):                 # noqa: N802 - Qt
        picker = self._picker()
        if picker is not None and hasattr(picker, "drag_to"):
            picker.drag_to(self.mapToGlobal(event.position().toPoint()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):              # noqa: N802 - Qt
        picker = self._picker()
        # BEFORE `super()`, which is what emits `clicked` and toggles the
        # button: a drag that has already painted the rectangle must not
        # then have its anchor flipped a second time by the click.
        dragged = (picker is not None and hasattr(picker, "finish_drag")
                   and picker.finish_drag())
        if dragged:
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _paint(self) -> None:
        colour = CHOSEN if self.isChecked() else EMPTY
        self.setStyleSheet(
            f"QPushButton {{ background: {colour}; border: 1px solid "
            f"rgba(255,255,255,0.18); border-radius: 3px; }}")


class PlateMapPicker(QDialog):
    """Display a plate map and return its selection as a well specification."""

    def __init__(self, value: str = "", layout: int = DEFAULT_LAYOUT,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Choose wells")
        self._layout_size = int(layout)
        self._wells = {}
        self._anchor: Optional[Tuple[int, int]] = None
        self._last_cell: Optional[Tuple[int, int]] = None
        self._before: Set[Tuple[int, int]] = set()
        self._adding = False
        self._dragged = False

        outer = QVBoxLayout(self)
        self._caption = QLabel("", self)
        self._caption.setObjectName("Muted")
        outer.addWidget(self._caption)

        self._holder = QWidget(self)
        self._grid = QGridLayout(self._holder)
        self._grid.setSpacing(2)
        area = QScrollArea(self)
        area.setWidgetResizable(True)
        area.setWidget(self._holder)
        outer.addWidget(area, 1)

        # THE THREE BUTTONS THE ASK NAMED, bottom right and in that order.
        row = QHBoxLayout()
        row.addStretch(1)
        self.plate_button = QPushButton("Plate", self)
        self.plate_button.setToolTip("Choose the plate layout.")
        self.plate_button.clicked.connect(lambda: self.ask_for_layout())
        row.addWidget(self.plate_button)
        buttons = QDialogButtonBox(self)
        self.done_button = buttons.addButton("Done",
                                             QDialogButtonBox.AcceptRole)
        self.close_button = buttons.addButton("Close",
                                              QDialogButtonBox.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        outer.addLayout(row)

        self.set_layout_size(self._layout_size, keep=value)

    # ------------------------------------------------------------ the grid

    def set_layout_size(self, layout: int, keep: str = "") -> None:
        """Rebuild the map for a plate layout and retain valid selections.

        Wells outside the new layout are dropped and counted in the caption,
        making a layout-induced selection change visible.
        """
        rows, columns = shape(layout)
        self._layout_size = int(layout)
        wanted = self.selection() if not keep else self._read(keep)

        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._wells = {}
        # A GRID KEEPS ITS ROW AND COLUMN COUNT when its items are taken out,
        # so the stretch that absorbed the spare space on the previous layout
        # would sit in the middle of a smaller one.
        for index in range(self._grid.rowCount()):
            self._grid.setRowStretch(index, 0)
        for index in range(self._grid.columnCount()):
            self._grid.setColumnStretch(index, 0)

        for column in range(1, columns + 1):
            self._grid.addWidget(QLabel(str(column), self._holder),
                                 0, column, Qt.AlignCenter)
        for row in range(1, rows + 1):
            self._grid.addWidget(QLabel(row_label(row), self._holder),
                                 row, 0, Qt.AlignCenter)
            for column in range(1, columns + 1):
                well = _Well(row, column, self._holder)
                well.pressed.connect(
                    lambda r=row, c=column: self._begin(r, c))
                well.toggled.connect(lambda *_: self._say())
                # CENTRED, LIKE ITS LABEL. Both share the cell, so a column
                # number is over its column and a row letter beside its row
                # however wide the cell has had to grow for the text in it.
                self._grid.addWidget(well, row, column, Qt.AlignCenter)
                self._wells[(row, column)] = well

        # The corner the labels live in is a cell of the plate too.
        self._grid.setColumnMinimumWidth(0, WELL_SIDE)
        self._grid.setRowMinimumHeight(0, WELL_SIDE)
        # WHERE THE SPARE SPACE GOES: past the last well, into an empty row
        # and column that hold nothing. The holder fills the scroll area, and
        # a grid with nowhere to put the extra width shares it out among the
        # cells -- which is exactly what pulls the numbers off their columns
        # and the letters off their rows as the window grows.
        self._grid.setRowStretch(rows + 1, 1)
        self._grid.setColumnStretch(columns + 1, 1)

        kept = {cell for cell in wanted if cell in self._wells}
        lost = len(wanted) - len(kept)
        self.set_selection(kept)
        self._say(lost)

    def ask_for_layout(self, chosen: Optional[int] = None) -> int:
        """Choose a supported plate size and return the active layout."""
        sizes = sorted(LAYOUTS)
        if chosen is None:
            current = sizes.index(self._layout_size) \
                if self._layout_size in sizes else sizes.index(DEFAULT_LAYOUT)
            text, ok = QInputDialog.getItem(
                self, "Plate layout", "How many wells?",
                [str(size) for size in sizes], current, False)
            if not ok:
                return self._layout_size
            chosen = int(text)
        self.set_layout_size(int(chosen))
        return self._layout_size

    # ------------------------------------------------------- the selection

    def _begin(self, row: int, column: int) -> None:
        """A press starts a drag; the anchor is where it started."""
        self._anchor = (row, column)

    # ------------------------------------------------------------ the drag
    #
    # `select_region` SELECTS A RECTANGLE WITHOUT A HUMAN, which is all it can
    # do on its own: press, move and release are what reach it from a mouse,
    # and a picker that is only ever driven through the method below has the
    # gesture implemented and unreachable.

    def begin_drag(self, row: int, column: int, modifiers=None) -> None:
        """Anchor a drag on one well.

        :param modifiers: the keyboard state at the press. With Ctrl the
            rectangle ADDS to what is already chosen; without it the drag
            REPLACES the selection, which is what every other grid in this
            application does.
        """
        self._anchor = (int(row), int(column))
        self._adding = bool(modifiers is not None
                            and (modifiers & Qt.ControlModifier))
        # WHAT TO GO BACK TO ON EVERY PREVIEW. A drag redraws from the state
        # at the PRESS rather than from the last frame, so growing and then
        # shrinking the rectangle leaves nothing behind.
        self._before = self.selection()
        self._dragged = False

    def well_at(self, position) -> Optional[Tuple[int, int]]:
        """The well under a GLOBAL point, or ``None``.

        Global, because the pointer is over a sibling of the widget that is
        receiving the events -- the pressed one keeps the grab.
        """
        for cell, well in self._wells.items():
            local = well.mapFromGlobal(position)
            if well.rect().contains(local):
                return cell
        return None

    def drag_to(self, position) -> None:
        """Preview the rectangle from the anchor to the well under ``position``.

        SHOWN WHILE DRAGGING, because a selection you cannot see until you
        let go is one you have to undo to correct.

        A MOVE INSIDE THE PRESSED WELL IS NOT A DRAG. Every real click
        carries a pixel or two of pointer travel, and drawing a one-well
        rectangle for it both clears the rest of the selection and leaves
        the release to fall through as an ordinary click -- which toggles
        the anchor straight back off, so a wobbled click lands nothing and
        takes the previous selection with it.

        AND ONCE IT IS A DRAG IT STAYS ONE. Sweeping away from the anchor
        and back again ends on the anchor, and treating that as no drag
        would hand the release to the click and toggle off the well the
        rectangle had just chosen.
        """
        if self._anchor is None:
            return
        cell = self.well_at(position)
        if cell is None or cell == getattr(self, "_last_cell", None):
            return
        if cell == self._anchor and not self._dragged:
            return
        self._last_cell = cell
        self._dragged = True
        self.set_selection(self._before if self._adding else set())
        self.select_region(self._anchor, cell, choosing=True)

    def finish_drag(self) -> bool:
        """End a drag. Returns whether one actually happened.

        `True` tells the well to swallow its release: the rectangle is
        already painted, and letting the click through would toggle the
        anchor a second time.
        """
        dragged = bool(getattr(self, "_dragged", False))
        self._anchor = None
        self._last_cell = None
        self._dragged = False
        if dragged:
            self._say()
        return dragged

    def select_region(self, start: Tuple[int, int], end: Tuple[int, int],
                      choosing: Optional[bool] = None) -> None:
        """Select or clear the rectangle between two plate coordinates.

        Parameters
        ----------
        start, end : tuple of int
            Inclusive one-based row and column coordinates.
        choosing : bool, optional
            ``True`` selects and ``False`` clears. ``None`` inverts the state
            of the starting well, matching spreadsheet-style drag selection.
        """
        top, bottom = sorted((int(start[0]), int(end[0])))
        left, right = sorted((int(start[1]), int(end[1])))
        if choosing is None:
            anchor = self._wells.get((int(start[0]), int(start[1])))
            choosing = not (anchor is not None and anchor.isChecked())
        for row in range(top, bottom + 1):
            for column in range(left, right + 1):
                well = self._wells.get((row, column))
                if well is not None and well.isChecked() != choosing:
                    well.setChecked(choosing)

    def selection(self) -> Set[Tuple[int, int]]:
        """Return selected wells as one-based ``(row, column)`` pairs."""
        return {cell for cell, well in self._wells.items()
                if well.isChecked()}

    def set_selection(self, cells) -> None:
        """Choose exactly ``cells`` and repaint the whole plate.

        :param cells: one-based ``(row, column)`` pairs, well-specification
            text such as ``"A01"`` or ``"c3"``, or a mixture. The field this
            picker edits is written in the second vocabulary, so it is worth
            answering to.

        THE REPAINT IS NOT OPTIONAL. Signals are blocked so that setting a
        hundred wells says the count once rather than a hundred times, and
        that also silences the ``toggled -> _paint`` connection -- so a well
        chosen without a click, which includes the picker's own starting
        value and everything kept across a layout change, would be checked
        and still drawn empty.
        """
        wanted: Set[Tuple[int, int]] = set()
        for cell in cells:
            if isinstance(cell, str):
                wanted |= self._read(cell)
            else:
                row, column = cell
                wanted.add((int(row), int(column)))
        for cell, well in self._wells.items():
            well.blockSignals(True)
            well.setChecked(cell in wanted)
            well.blockSignals(False)
            well._paint()
        self._say()

    def _read(self, text) -> Set[Tuple[int, int]]:
        try:
            return parse(text, self._layout_size)
        except WellSpecError:
            # A FIELD THAT WILL NOT PARSE OPENS EMPTY rather than refusing to
            # open: the picker is how a user fixes a value they typed wrong.
            return set()

    def value(self) -> str:
        """Return the selection serialized for a well-specification field."""
        return to_text(self.selection(), self._layout_size)

    def _say(self, lost: int = 0) -> None:
        chosen = len(self.selection())
        rows, columns = shape(self._layout_size)
        note = (f"{self._layout_size}-well plate ({rows} x {columns}) — "
                f"{chosen} well(s) chosen")
        if lost:
            note += (f". {lost} well(s) from the previous selection are not "
                     f"on this layout and were dropped")
        self._caption.setText(note)
