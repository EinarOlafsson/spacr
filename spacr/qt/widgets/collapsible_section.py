"""A titled section that folds away in tabs containing several panels.

Resizable sections let users give the active panel more room, while folding
inactive sections prevents controls from competing for the same vertical
space.

FOLDED, NOT REMOVED, and the header stays put: a section that disappeared
would take with it the only clue that the feature exists, which is the same
rule the greyed-out picture settings follow.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QSizePolicy, QToolButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """``content`` under a header that folds it away.

    :param title: what the header says. Kept short -- it is a name, not a
        description; the panel inside says what it does.
    :param content: the widget to fold. Reparented here.
    :param expanded: whether it starts open.
    :param parent: parent widget; ownership only.
    """

    toggled = Signal(bool)

    #: How tall the section is when folded: the header and nothing else.
    #: Read by the splitter, which otherwise keeps a minimum that would stop
    #: a folded section from actually getting out of the way.
    FOLDED_HEIGHT = 26

    def __init__(self, title: str, content: QWidget, *, expanded: bool = True,
                 parent=None):
        super().__init__(parent)
        self._title = str(title)
        self._content = content

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._header = QToolButton(self)
        self._header.setText(self._title)
        self._header.setCheckable(True)
        self._header.setChecked(bool(expanded))
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._header.setAutoRaise(True)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._header.setToolTip(f"Fold {self._title} away, or open it again")
        # GRAY, TRANSPARENT, ROUNDED, BLUE ON HOVER. Asked for 2026-08-19:
        # "the black dropdowns should be gray and they should have rounded
        # edges when hovered blue, and they should be transparent". A
        # QToolButton with no rule of its own paints the palette's button
        # colour as an opaque block, which is the "black categories" in both
        # this panel and the measure tab.
        # AND THE RESTING HEADING IS THE THEME'S FOREGROUND (198). Asked for
        # 2026-08-21: "the text in the measurements tab for the sub
        # categories should be white when they are not highlighted (in dark
        # mode oposite in bright mode)."
        #
        # It was `palette(mid)` at rest and `palette(text)` only when open --
        # which is backwards. THE UNHIGHLIGHTED STATE IS THE ONE A USER
        # READS: on a tab with four folded sections at most one is open, and
        # the rest are what they are scanning to decide where to go. Dimming
        # them says "secondary" about the only thing on screen that is not.
        #
        # `palette(text)`, NEVER A LITERAL. `#FFFFFF` here is the same fault
        # instruction 178 removed from eleven figure call sites: it reads on
        # one theme and vanishes on the other, and the author sees only the
        # one they use. The palette answers "white or near-black" for
        # whichever theme is live.
        #
        # THE HIGHLIGHT IS STILL VISIBLE, and that is the condition on this
        # change: hover keeps its blue wash and the fold arrow still turns.
        # What no longer distinguishes the states is the text going away.
        self._header.setStyleSheet(
            "QToolButton {"
            "  background: transparent;"
            "  border: none;"
            "  border-radius: 6px;"
            "  padding: 3px 6px;"
            "  text-align: left;"
            "  color: palette(text);"
            "}"
            "QToolButton:hover {"
            "  background: rgba(45, 119, 188, 0.18);"
            "  color: palette(highlight);"
            "}"
            "QToolButton:checked { color: palette(text); }")
        self._header.toggled.connect(self._apply)
        layout.addWidget(self._header)

        content.setParent(self)
        layout.addWidget(content, 1)

        # The minimum the CONTENT wants, remembered before anything folds it.
        # Restoring the section has to put back the height that made the
        # panel usable, and once folded that number is no longer readable off
        # the widget.
        self._open_minimum = max(content.minimumHeight(), 0)
        self._apply(bool(expanded))

    # ------------------------------------------------------------- folding

    def is_expanded(self) -> bool:
        """Whether the body is showing.

        READ OFF THE HEADER, not a flag beside it. The header button IS the
        state, so a cached copy could disagree with what the user sees.

        :returns: True when open.
        """
        return self._header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        """Open or close the section.

        :param expanded: True to show the body.
        """
        self._header.setChecked(bool(expanded))

    def title(self) -> str:
        """The section's caption, as written rather than as displayed.

        The header uppercases it and may append a maturity badge; this is
        the string a catalog or a settings file is keyed on.

        :returns: the title.
        """
        return self._title

    def content(self) -> QWidget:
        """The widget this section wraps.

        :returns: the body widget.
        """
        return self._content

    def set_open_minimum(self, height: int) -> None:
        """How short the section may be dragged while it is open."""
        self._open_minimum = max(int(height), 0)
        if self.is_expanded():
            self.setMinimumHeight(self._open_minimum + self.FOLDED_HEIGHT)

    def _apply(self, expanded: bool) -> None:
        self._header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self._content.setVisible(bool(expanded))
        if expanded:
            self.setMinimumHeight(self._open_minimum + self.FOLDED_HEIGHT)
            self.setMaximumHeight(16777215)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        else:
            # BOTH BOUNDS. A minimum alone leaves the splitter free to hand
            # the folded section the space it just gave up, which looks like
            # the fold did nothing.
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            self.setMinimumHeight(self.FOLDED_HEIGHT)
            self.setMaximumHeight(self.FOLDED_HEIGHT)
        self.toggled.emit(bool(expanded))
