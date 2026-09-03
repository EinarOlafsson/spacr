"""A left-to-right layout that wraps, and the widget that hosts one.

Lifted out of `spacr.qt.screens.settings_model`, where it was written for
the settings panel's chip strip, because a wrapping row is not a settings
idea: the regression results header wants one too. Three combo boxes with
minimum widths, a run name of unpredictable length and a QHBoxLayout do not
fit in a narrow panel, and Qt's answer to a box it cannot satisfy is to let
the children overlap -- measured at 577 px, where the second box started
48 px inside the first and the third ran 32 px off the panel.
"""
from __future__ import annotations

from typing import Any, List

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtWidgets import QLayout, QSizePolicy, QWidget


class FlowLayout(QLayout):
    """A left-to-right layout that wraps onto a new line when it runs out.

    Chips have to wrap: ``controls`` ships thirty of them and a horizontal
    box would either clip them or force the settings panel wider than the
    window.

    :param parent: parent widget.
    :param spacing: pixels between chips, horizontally and vertically.
    """

    def __init__(self, parent=None, spacing: int = 4):
        super().__init__(parent)
        self._items: List[Any] = []
        self._space = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:            # noqa: N802 (Qt override)
        """Append a layout item (Qt calls this for every added widget)."""
        self._items.append(item)

    def count(self) -> int:
        """Number of items in the layout."""
        return len(self._items)

    def itemAt(self, index):                    # noqa: N802 (Qt override)
        """Return the item at ``index``, or None when out of range."""
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):                    # noqa: N802 (Qt override)
        """Remove and return the item at ``index``, or None."""
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):              # noqa: N802 (Qt override)
        """Never ask for extra space in either direction."""
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:        # noqa: N802 (Qt override)
        """Height depends on width -- that is the whole point of wrapping."""
        return True

    def heightForWidth(self, width: int) -> int:    # noqa: N802 (Qt override)
        """Height needed to lay the chips out inside ``width``."""
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect) -> None:        # noqa: N802 (Qt override)
        """Place every chip inside ``rect``."""
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:                # noqa: N802 (Qt override)
        """Preferred size -- the minimum, since the height is width-driven."""
        return self.minimumSize()

    def minimumSize(self) -> QSize:             # noqa: N802 (Qt override)
        """The largest single chip, plus margins."""
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _do_layout(self, rect, test_only: bool) -> int:
        margins = self.contentsMargins()
        area = rect.adjusted(margins.left(), margins.top(),
                             -margins.right(), -margins.bottom())
        x, y, line_height = area.x(), area.y(), 0
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + self._space
            if next_x - self._space > area.right() and line_height > 0:
                x = area.x()
                y = y + line_height + self._space
                next_x = x + hint.width() + self._space
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


class FlowHost(QWidget):
    """The widget a :class:`FlowLayout` lives in.

    Qt only consults a layout's ``heightForWidth`` through the widget that
    owns it, and only when that widget's size policy says its height depends
    on its width. Without this the strip reported a one-line height however
    many chips it held, and ``controls`` (thirty of them) drew off the edge
    of the settings column instead of wrapping.

    :param parent: parent widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        policy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

    def hasHeightForWidth(self) -> bool:      # noqa: N802 (Qt override)
        """Yes -- more width means fewer rows of chips."""
        return True

    def heightForWidth(self, width: int) -> int:   # noqa: N802 (Qt override)
        """Height the chips need once wrapped into ``width``."""
        layout = self.layout()
        if layout is None:
            return super().heightForWidth(width)
        return layout.heightForWidth(width)

    def sizeHint(self) -> QSize:              # noqa: N802 (Qt override)
        """Preferred size at the current width, so the row grows as chips
        are added rather than clipping them."""
        layout = self.layout()
        if layout is None:
            return super().sizeHint()
        width = max(self.width(), layout.minimumSize().width())
        return QSize(width, layout.heightForWidth(width))
