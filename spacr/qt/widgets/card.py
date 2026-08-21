"""
Card — QFrame with rounded border, optional title bar, and a body widget.

Consumers add content to `card.body_layout` (a QVBoxLayout).
"""
from __future__ import annotations

from PySide6.QtWidgets import QFrame, QVBoxLayout, QLabel, QWidget

from ..theme import SPACING


class Card(QFrame):
    """Rounded-border container with optional title/subtitle and a body area.

    Consumers add content to :attr:`body_layout`.

    :param title: optional heading rendered above the body.
    :param subtitle: optional muted subheading rendered under the title.
    :ivar body: inner QWidget that holds the body layout.
    :ivar body_layout: QVBoxLayout consumers add widgets to.
    """

    def __init__(self, title: str = "", subtitle: str = "", parent=None,
                 *, foldable: bool = False, fold_key: str = ""):
        """Initialize the card and optional persistent folding behavior.

        :param foldable: allow the title to hide or restore the body. Disabled
            by default because folding is useful only when adjacent content
            can occupy the released space.
        :param fold_key: ``"<module>/<panel>"``; given, the fold survives a
            restart.
        """
        super().__init__(parent)
        self.setObjectName("Card")
        #: The :class:`~spacr.qt.widgets.foldable.Folder`, or ``None``. HELD,
        #: because it owns the event filter and one nobody keeps stops
        #: working silently.
        self.folder = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["md"], SPACING["md"], SPACING["md"], SPACING["md"])
        outer.setSpacing(SPACING["sm"])

        title_label = None
        if title:
            title_label = QLabel(title)
            title_label.setObjectName("CardTitle")
            outer.addWidget(title_label)
        self.title_label = title_label
        if subtitle:
            sub_label = QLabel(subtitle)
            sub_label.setObjectName("CardSubtitle")
            sub_label.setWordWrap(True)
            outer.addWidget(sub_label)

        if title or subtitle:
            divider = QFrame()
            divider.setObjectName("Divider")
            divider.setFrameShape(QFrame.HLine)
            outer.addWidget(divider)

        self.body = QWidget(self)
        # The global `QWidget { background: bg }` rule would paint the body
        # solid black over the card's rounded surface. Make it transparent so
        # the card colour shows behind the content (bars, etc.).
        self.body.setObjectName("CardBody")
        self.body.setStyleSheet("QWidget#CardBody { background: transparent; }")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(SPACING["sm"])
        outer.addWidget(self.body, 1)

        if foldable and title_label is not None:
            from .foldable import make_foldable

            # The BODY folds, not the card: the title has to stay to be
            # clicked again, which is what makes the folded state a strip
            # that names itself rather than a disappearance.
            self.folder = make_foldable(title_label, self.body, name=title,
                                        persist_key=fold_key)
