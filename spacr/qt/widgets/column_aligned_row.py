"""A row of buttons laid out over the columns of the table underneath it.

Regression's Input Tables category starts with a Download row -- Score,
Count, Measurements (.db), Image crops -- and the first three of those fill
one column each of the paired-data table directly below. Each button is
centred on its column, and the download table's column widths are locked to
the widths of their counterparts in the paired-data table below.

ONE COLUMN MODEL, READ IN ONE DIRECTION. The build the request describes --
a second table above the first, both told to keep the same widths -- is two
column models that have to be kept in step, and this repository has already
paid for that once (see the `_ClusterSettingsDialog` docstring on what two
editors of one setting cost). This layout owns no widths at all. It asks the
paired table's ``QHeaderView`` where each section starts and how wide it is,
every time it lays out, and it writes nothing back: the header never learns
that a strip is following it, so there is nothing to drift.

WHY A QLayout RATHER THAN SPACERS IN A QHBoxLayout. Stretches and fixed
spacers can be tuned to line up at one column width and are wrong at every
other one, so each drag of a column edge would mean rebuilding them. A
layout's ``setGeometry`` runs on every relayout anyway, so reading the header
there is both shorter and correct at any width; all that is left to arrange
is that a column resize causes a relayout, which is what the ``sectionResized``
connection below is for.

WHY A BUTTON IS CLAMPED TO ITS COLUMN. Measured on a 1400x900 screen with the
table's default 100 px columns: "Measurements (.db)" wants 127 px, so left at
its natural width and centred on its column it would reach 3.5 px into the
Count button beside it. Two download buttons drawn overlapping is exactly the
confusion the alignment exists to remove, so the button is given its column
and no more -- which is also what "lock the width ... to the width of the
paired data column counterparts below" asks for. The cost is that a label
longer than its column is clipped by Qt at the default width; the full
sentence is in the button's tooltip, and widening the column shows it.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

import shiboken6
from PySide6.QtCore import QEvent, QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QWidget, QWidgetItem

__all__ = ["ColumnAlignedRow", "align_row_to_columns", "TRAILING_SPACING"]

#: Gap left between two buttons that have no column to sit over, in px.
#:
#: Small and fixed rather than taken from ``SPACING``: these are the leftovers
#: of the row -- a button whose download fills a SETTING rather than a table
#: column -- and they are placed as a plain run after the last column, not as
#: part of the aligned grid.
TRAILING_SPACING = 6


def _alive(obj) -> bool:
    """True when ``obj``'s C++ half is still there.

    The strip outlives nothing, but the header can go first: a closed screen
    deletes the table while the layout is still connected to its signals, and
    ``sectionSize`` on a deleted header is a hard crash rather than an
    exception.
    """
    try:
        return obj is not None and shiboken6.isValid(obj)
    except Exception:                                        # noqa: BLE001
        return obj is not None


class ColumnAlignedRow(QLayout):
    """Lay a row of widgets out over the sections of a ``QHeaderView``.

    Each managed widget carries either a column index -- it is centred in
    that column's span, clamped to its width -- or ``None``, in which case it
    is placed in a plain left-to-right run after the last aligned column.

    The header is read, never written. Nothing here calls ``resizeSection``
    or a resize mode, so the user's own column widths remain the only source
    of truth for what a column is.
    """

    def __init__(self, header, parent: Optional[QWidget] = None) -> None:
        # `parent` installs this as the widget's layout, which is why the
        # caller has to have removed the previous one first.
        super().__init__(parent)
        self._items: list[Tuple[QWidgetItem, Optional[int]]] = []
        self._header = header
        self.setContentsMargins(0, 0, 0, 0)
        if _alive(header):
            # THE ONLY THREE THINGS THAT MOVE A COLUMN. A drag on a section
            # edge (`sectionResized`), a reorder if one is ever enabled
            # (`sectionMoved`), and the header itself being re-laid-out when
            # the table changes width (`geometriesChanged`). None of them
            # resizes the strip, so without these the buttons would stay
            # where the last strip resize left them.
            header.sectionResized.connect(self._restate)
            header.sectionMoved.connect(self._restate)
            header.geometriesChanged.connect(self._restate)
            # And the table MOVING under the strip, which changes where the
            # columns are without changing their widths -- a section opening
            # above it does exactly that. The viewport is watched rather than
            # the table because it is the widget the column positions are
            # measured from.
            header.viewport().installEventFilter(self)

    # ------------------------------------------------------ QLayout's five

    def addItem(self, item) -> None:                          # noqa: N802
        """Qt's own door, used by ``addWidget``: no column, so it trails."""
        self._items.append((item, None))

    def add_over_column(self, widget: QWidget,
                        column: Optional[int]) -> None:
        """Manage ``widget``, centred over ``column`` (``None`` to trail).

        Separate from ``addWidget`` because Qt's signature has no room for
        the column, and a column set afterwards through a second call would
        be a second place the pairing is written down.
        """
        if widget is None or not _alive(widget):
            return
        widget.setParent(self.parentWidget())
        self._items.append((QWidgetItem(widget), column))
        self.invalidate()

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):                                  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index][0]
        return None

    def takeAt(self, index):                                  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)[0]
        return None

    def sizeHint(self) -> QSize:                              # noqa: N802
        """Wide enough for the buttons side by side, tall enough for one.

        Not the header's length: the row must remain usable in a form that is
        narrower than the table's columns add up to, and the columns are the
        table's business rather than the strip's.
        """
        width = 0
        height = 0
        for item, _column in self._items:
            hint = item.sizeHint()
            width += hint.width() + TRAILING_SPACING
            height = max(height, hint.height())
        return QSize(width, height)

    def minimumSize(self) -> QSize:                           # noqa: N802
        """Zero wide. A strip narrower than its buttons still shows them in
        the right place, because the places come from the table, not from the
        strip's own width."""
        height = 0
        for item, _column in self._items:
            height = max(height, item.minimumSize().height())
        return QSize(0, height)

    def expandingDirections(self):                            # noqa: N802
        """Horizontally only. QLayout's default claims both, which makes the
        form give this one-button-high row every spare pixel of height."""
        return Qt.Orientations(Qt.Horizontal)

    def setGeometry(self, rect: QRect) -> None:               # noqa: N802
        """Put each widget over its column, and the rest after them."""
        super().setGeometry(rect)
        owner = self.parentWidget()
        if owner is None:
            return
        trailing = []
        # Where the aligned run ends, so the un-aligned buttons start clear
        # of it. Starts at the strip's own left edge for the case where no
        # column could be read at all.
        right = rect.x()
        for item, column in self._items:
            hint = item.sizeHint()
            height = min(hint.height(), rect.height())
            top = rect.y() + (rect.height() - height) // 2
            span = self._column_span(owner, column)
            if span is None:
                trailing.append((item, hint, height, top))
                continue
            left, width = span
            drawn = min(hint.width(), width)
            item.setGeometry(QRect(left + (width - drawn) // 2, top,
                                   drawn, height))
            right = max(right, left + width)
        for item, hint, height, top in trailing:
            right += TRAILING_SPACING
            item.setGeometry(QRect(right, top, hint.width(), height))
            right += hint.width()

    # ------------------------------------------------------------- reading

    def _column_span(self, owner: QWidget,
                     column: Optional[int]):
        """``(left, width)`` of ``column`` in ``owner``'s coordinates.

        ``None`` when there is no column to follow -- the widget was given
        none, the header has gone, the index is past its end, or the column
        is hidden (width 0). A hidden column must not orphan the button above
        it, so the button joins the trailing run instead of being drawn on
        top of its neighbour.

        Mapped THROUGH GLOBAL COORDINATES rather than through a common
        ancestor: the strip and the table are two field widgets of the same
        QFormLayout today, but nothing here should depend on that, and
        ``mapFrom`` requires the ancestor relationship that global mapping
        does not.
        """
        header = self._header
        if column is None or not _alive(header):
            return None
        if column < 0 or column >= header.count():
            return None
        width = header.sectionSize(column)
        if width <= 0:
            return None
        viewport = header.viewport()
        if not _alive(viewport):
            return None
        edge = viewport.mapToGlobal(
            QPoint(header.sectionViewportPosition(column), 0)).x()
        return owner.mapFromGlobal(QPoint(edge, 0)).x(), width

    def _restate(self, *_args) -> None:
        """Lay out again because the columns moved."""
        if self.parentWidget() is None:
            return
        self.invalidate()
        self.activate()

    def eventFilter(self, watched, event):                    # noqa: N802
        """Follow the table when it moves or changes size."""
        if event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
            self._restate()
        return False


def align_row_to_columns(
        strip: QWidget, header,
        columns: Iterable[Tuple[QWidget, Optional[int]]],
) -> Optional[ColumnAlignedRow]:
    """Re-lay ``strip``'s widgets out over ``header``'s sections.

    Idempotent: a strip that is already following this header is returned
    unchanged, so a repeated show costs nothing and cannot stack two layouts.

    :param strip: the widget holding the buttons. Its existing layout is
        emptied and destroyed -- the widgets survive, reparented on ``strip``.
    :param header: the ``QHeaderView`` whose columns are followed.
    :param columns: ``(widget, column index or None)`` in the order the
        un-aligned ones should trail in.
    :returns: the installed layout, or ``None`` when there was nothing to do.
    """
    if strip is None or not _alive(strip) or not _alive(header):
        return None
    existing = strip.layout()
    if isinstance(existing, ColumnAlignedRow):
        return existing
    if existing is not None:
        # EMPTIED FIRST, THEN DELETED. Deleting a layout that still holds its
        # items deletes the buttons with it, which would take the Download
        # row away rather than align it.
        while existing.count():
            item = existing.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.setParent(strip)
        # `shiboken6.delete`, not `deleteLater`: the new layout is installed
        # on the next line, and Qt refuses to install one while the widget
        # still has a layout -- which a deferred deletion leaves it with.
        shiboken6.delete(existing)
    row = ColumnAlignedRow(header, strip)
    for widget, column in columns:
        row.add_over_column(widget, column)
    row.activate()
    return row
