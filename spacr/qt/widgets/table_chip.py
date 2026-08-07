"""A removable chip naming one table in the Gate Editor's working set.

The working set is a combination, not a selection: picking nucleus adds
nuclear measurements alongside the cell ones rather than replacing them. A
combination needs to be VISIBLE -- otherwise the only way to know what is
merged is to read the axis picker and infer it -- so each member gets a chip
with an x.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QWidget

from ..theme import SPACING, active_palette, register_widget_qss

#: The key this module's stylesheet is registered under.
QSS_NAME = "TableChip"


def _chip_qss(palette, opacity=None) -> str:
    """Blue rounded box, per the request, in the theme's own accent.

    The accent rather than a literal blue: a hard-coded colour is wrong in
    whichever theme was not being looked at when it was written.
    """
    # The theme's own text colour: white on the dark themes, as asked, and
    # dark on the light ones without a second rule. It was the WINDOW colour
    # before, which is black on dark -- black on a blue chip.
    #
    # Contrast maths would pick black here (a mid blue is bright enough that
    # black scores higher), so it is deliberately not used: the ask was white
    # on dark, and following the theme's text colour is what keeps that true
    # in every theme rather than only this one.
    ink = palette["fg"]
    return f"""
    QWidget#TableChip {{
        background: {palette['accent']};
        border-radius: 9px;
    }}
    QLabel#TableChipName {{
        color: {ink};
        background: transparent;
        padding: 1px 2px 1px 8px;
    }}
    QPushButton#TableChipClose {{
        color: {ink};
        background: transparent;
        border: none;
        padding: 0px 6px 2px 4px;
        font-weight: 600;
    }}
    QPushButton#TableChipClose:hover {{
        color: {palette['warning']};
    }}
    """


register_widget_qss(QSS_NAME, _chip_qss, replace=True)


class TableChip(QWidget):
    """One table in the working set. Emits :attr:`removed` with its name."""

    removed = Signal(str)

    def __init__(self, name: str, parent=None, *, removable: bool = True):
        super().__init__(parent)
        self.setObjectName("TableChip")
        self._name = name

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        label = QLabel(name, self)
        label.setObjectName("TableChipName")
        row.addWidget(label)

        self._close = QPushButton("×", self)
        self._close.setObjectName("TableChipClose")
        self._close.setCursor(Qt.PointingHandCursor)
        self._close.setToolTip(f"Remove {name} from the working set")
        self._close.clicked.connect(lambda: self.removed.emit(self._name))
        # The last table has no x: a gate editor with no table is a screen
        # with nothing on it, and the user's next move would be to load the
        # same table again.
        self._close.setVisible(removable)
        row.addWidget(self._close)

        metrics = self.fontMetrics()
        self.setMinimumHeight(metrics.height() + 6)
        self.setMinimumWidth(metrics.horizontalAdvance(name) + 34)

    @property
    def name(self) -> str:
        return self._name
