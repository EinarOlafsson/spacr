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
        :param parent: parent widget; ownership only.
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
        self.body.setObjectName("CardBody")
        self.body.setStyleSheet("QWidget#CardBody { background: transparent; }")
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(SPACING["sm"])
        outer.addWidget(self.body, 1)

        if foldable and title_label is not None:
            from .foldable import make_foldable

            self.folder = make_foldable(title_label, self.body, name=title,
                                        persist_key=fold_key)
        self._outer = outer
        self._title_row = None
        if self.folder is not None:
            self.follow_fold(self.folder)

    def follow_fold(self, folder) -> None:
        """Let the card shrink to its heading while ``folder`` is shut.

        The body is laid out with a stretch, and a layout counts that stretch
        even while the body is hidden, so a folded card still asked for room
        and shared it evenly with the stretch that locks a folded heading to
        the bottom of its pane (item 515): the folded Console came out half
        the pane tall. Folded, the body gives its stretch up; opened, it takes
        it back.

        :param folder: the :class:`~spacr.qt.widgets.foldable.Folder` that
            folds :attr:`body`.
        """
        def apply(shut: bool, _by_user: bool = True) -> None:
            """Match the body's stretch to the fold."""
            self._outer.setStretchFactor(self.body, 0 if shut else 1)

        folder.add_listener(apply)
        apply(folder.shut)

    def add_title_action(self, widget: QWidget) -> None:
        """Put ``widget`` at the right-hand end of the title row.

        The title row is built on first use, so a card nobody adds an action
        to keeps its original layout exactly.

        :param widget: the control to add, e.g. a Refresh button.
        """
        if self._title_row is None:
            from PySide6.QtWidgets import QHBoxLayout

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(SPACING["sm"])
            if self.title_label is not None:
                index = self._outer.indexOf(self.title_label)
                self._outer.removeWidget(self.title_label)
                row.addWidget(self.title_label)
                row.addStretch(1)
                self._outer.insertLayout(max(index, 0), row)
            else:
                row.addStretch(1)
                self._outer.insertLayout(0, row)
            self._title_row = row
        self._title_row.addWidget(widget)


class _CardBuiltWhenShown(Card):
    """A :class:`Card` whose body can wait until the card is first shown.

    A module screen is polished widget by widget against the whole
    stylesheet when it opens, so every widget on it costs time whether or
    not anybody can see it. A card that starts hidden -- a results panel
    before there are results, a search panel behind its switch -- is the
    largest share of that on the heavy screens, and nobody sees its body
    until the card is shown. A plain ``Card`` in every respect otherwise,
    ``Card`` rules in the stylesheet included.
    """

    def __init__(self, *args, **kwargs):
        """Build the card with nothing waiting to fill it."""
        super().__init__(*args, **kwargs)
        self._build_body = None

    def build_body_when_first_shown(self, build) -> None:
        """Defer filling this card's body until the card is first shown.

        ``build`` runs ONCE, before the card becomes visible, from
        :meth:`setVisible`; :meth:`showEvent` is the belt to that brace for
        a card that is revealed by its parent rather than shown itself. The
        caller may also run it earlier through :meth:`ensure_body_built`,
        which is what a screen does when code asks for the panel inside.

        :param build: a callable with no arguments that fills
            :attr:`body_layout`.
        """
        self._build_body = build

    def body_is_built(self) -> bool:
        """Whether no deferred body is still waiting to be built."""
        return self._build_body is None

    def ensure_body_built(self) -> bool:
        """Build the deferred body now, if one is waiting.

        :returns: ``True`` when this call built it.
        """
        build, self._build_body = self._build_body, None
        if build is None:
            return False
        build()
        return True

    def setVisible(self, visible: bool) -> None:                 # noqa: N802
        """Build a deferred body before the card can be seen.

        Before, not after: children added to a widget that is already
        visible are only shown on a later turn of the event loop, so a body
        built from the show would leave one frame of an empty card.
        """
        if visible and self._build_body is not None:
            self.ensure_body_built()
        super().setVisible(visible)

    def showEvent(self, event) -> None:                          # noqa: N802
        """Build a deferred body that a parent's show revealed."""
        if self._build_body is not None:
            self.ensure_body_built()
        super().showEvent(event)
