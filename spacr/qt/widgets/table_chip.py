"""Display each table in the Gate Editor's working set as a removable chip.

The working set is additive: selecting a nucleus table keeps existing cell
measurements instead of replacing them. One chip per member makes the pending
merge explicit and provides a direct removal control.
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ..theme import close_mark_button, register_widget_qss

#: The key this module's stylesheet is registered under.
QSS_NAME = "TableChip"


def _chip_qss(palette, opacity=None) -> str:
    """Return rounded chip styling using the supplied theme accent."""
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
    """


register_widget_qss(QSS_NAME, _chip_qss, replace=True)


class TableChip(QWidget):
    """One table in the working set. Emits :attr:`removed` with its name.

    :param name: the table's name. It is both the caption and the payload of
        :attr:`removed`, so it has to be the name the working set keys on
        rather than anything prettied up for display.
    :param parent: parent widget.
    :param removable: whether the chip offers its remove button. The Gate
        Editor passes False for the LAST table in the set, so the working set
        can never be emptied to nothing.
    """

    removed = Signal(str)

    def __init__(self, name: str, parent=None, *, removable: bool = True):
        """Build one removable chip naming a table or a source.

        :param name: what the chip stands for; also its label.
        :param parent: parent widget, or ``None``.
        :param removable: show the close mark. Set ``False`` for a chip the set
            cannot go below.
        """
        super().__init__(parent)
        self.setObjectName("TableChip")
        self._name = name

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)

        label = QLabel(name, self)
        label.setObjectName("TableChipName")
        row.addWidget(label)

        self._close = close_mark_button(
            self, tooltip=f"Remove {name} from the working set")
        self._close.setObjectName("TableChipClose")
        self._close.clicked.connect(lambda: self.removed.emit(self._name))
        self._close.setVisible(removable)
        row.addWidget(self._close)

        self.ensurePolished()
        metrics = self.fontMetrics()
        self.setMinimumHeight(
            max(metrics.height() + 6, self._close.height() + 4))
        self.setMinimumWidth(
            metrics.horizontalAdvance(name) + 12 + self._close.width())

    @property
    def name(self) -> str:
        """The table this chip stands for.

        :returns: the table's name.
        """
        return self._name
