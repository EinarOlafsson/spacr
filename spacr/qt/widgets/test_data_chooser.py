"""A popup that explains each test-data route before either is started.

Instruction of 2026-09-01, for Annotate and then Classify: one "Load test
data" button opening a dialog with two buttons, a description that fills in on
hover, and a Close button.

It lives here rather than on a screen because two screens now use it and a
third is asked for. It fetches NOTHING itself -- it reports the chosen route
and the screen does the work, which keeps it testable without a network.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, QRect, Qt
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..i18n import tr


class TestDataChooser(QDialog):
    """Pick which half of the example plate to fetch, and read why first.

    The two routes need different halves of the plate and differ in size, and
    the old pair of buttons named the choice ("crops" / "streaming") without
    room to explain it -- the difference lived in a tooltip that had to be
    hunted for. Here the description sits UNDER the buttons and fills in on
    hover, so both routes can be compared before either is started. A download
    of several hundred megabytes deserves that much.

    Nothing is fetched by this dialog. It reports the chosen route and the
    screen does the work, so the dialog stays testable without a network.

    :param parent: parent widget.
    """

    #: ``key -> (button text, what it fetches)``. The sizes are the real
    #: archive sizes, and they are in the text because "about 280 MB" is the
    #: fact a user on a hotel connection is actually deciding on.
    ROUTES = (
        ("load", "Load",
         "Download about 280 MB: 2,341 single-cell crops already cut, with "
         "the measurements database that indexes them, labelled infected or "
         "not.\n\nFor annotating images that already exist on disk. The "
         "source folder and the annotation settings are filled in with it, "
         "and Image source is set to LOAD IMAGES."),
        ("stream", "Stream",
         "Download about 390 MB: the merged arrays the crops were cut from, "
         "so a set can be streamed as the page is drawn rather than read off "
         "disk.\n\nNeeds no exported crops. Unpacks into the same plate "
         "folder as Load, so pressing both leaves a complete plate and either "
         "route then works. Streaming reads where each cell sits from the "
         "measurements database, so the Load half is fetched too when it is "
         "not on disk yet. Image source is set to STREAM IMAGES."),
    )

    #: What the description pane says before anything is hovered.
    RESTING_TEXT = ("Hover a button to see what it downloads and what it "
                    "sets. Nothing is fetched until you press one.")

    #: The width the dialog opens at, and the width the pane is measured at.
    #:
    #: A DIALOG THIS SIZE IS DECIDED BY ITS TEXT, and the text is two
    #: paragraphs. Left to Qt the buttons set the width -- "Load" and
    #: "Stream" are short, so the dialog came out 282 px wide and the
    #: descriptions became a tall thin ribbon: 187 px of pane for 316
    #: characters, in a window 509 px tall. At this width each route is a
    #: few lines and the whole dialog is under half that.
    #:
    #: Reported as the load-test-data window in Annotate opening far too tall; make it be as small as possible while still
    #: fitting the text".
    DIALOG_WIDTH = 460

    #: Buttons per row, or 0 for one row. A chooser with two routes wants
    #: them side by side; Import's has two dozen, and a single row of those
    #: is wider than the screen.
    COLUMNS = 0

    def __init__(self, parent=None):
        """Build the test-data chooser.

        The minimum width is set before the description pane is measured: the
        measurement asks how tall the longest description is AT A GIVEN WIDTH,
        and measuring at one width while displaying at another is what left the
        pane sized for a column it never had.

        :param parent: parent widget, or ``None``.
        """
        super().__init__(parent)
        self.setWindowTitle(tr("Load test data"))
        self.chosen = ""
        self.setMinimumWidth(self.DIALOG_WIDTH)
        #: Whether a layout pass has given the pane a real width. Until one
        #: has, the pane reports the 100 px every freshly constructed widget
        #: reports, which is not a width it will ever be drawn at.
        self._laid_out = False

        layout = QVBoxLayout(self)

        buttons = QGridLayout() if self.COLUMNS > 0 else QHBoxLayout()
        self._buttons = {}
        for position, (key, label, description) in enumerate(self.ROUTES):
            button = QPushButton(tr(label), self)
            button.setCursor(Qt.PointingHandCursor)
            button.setProperty("routeKey", key)
            button.setToolTip(tr(description))
            button.installEventFilter(self)
            button.clicked.connect(
                lambda checked=False, k=key: self._choose(k))
            if self.COLUMNS > 0:
                buttons.addWidget(button, position // self.COLUMNS,
                                  position % self.COLUMNS)
            else:
                buttons.addWidget(button)
            self._buttons[key] = button
        layout.addLayout(buttons)

        self._description = QLabel(tr(self.RESTING_TEXT), self)
        self._description.setWordWrap(True)
        self._description.setObjectName("TestDataDescription")
        self._size_the_description_pane()
        self._description.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self._description)

        closing = QHBoxLayout()
        closing.addStretch(1)
        close = QPushButton(tr("Close"), self)
        close.clicked.connect(self.reject)
        closing.addWidget(close)
        layout.addLayout(closing)

        self.adjustSize()

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        """Fill the pane on hover, and empty it on leave."""
        kind = event.type()
        if kind == QEvent.Enter:
            key = str(watched.property("routeKey") or "")
            for route_key, _label, description in self.ROUTES:
                if route_key == key:
                    self._description.setText(tr(description))
                    break
        elif kind == QEvent.Leave:
            self._description.setText(tr(self.RESTING_TEXT))
        return super().eventFilter(watched, event)

    def description_text(self) -> str:
        """What the pane currently says. For tests and the accessibility tree."""
        return self._description.text()

    def every_description(self) -> tuple:
        """Every string the pane can ever show, resting text included.

        Public because it is what a clipping check wants to iterate: any
        sweep asking whether text fits its container needs exactly this
        list, and building a second one risks measuring a different set
        of strings than the pane can actually display.
        """
        return (tr(self.RESTING_TEXT),) + tuple(
            tr(description) for _key, _label, description in self.ROUTES)

    def _size_the_description_pane(self) -> None:
        """Make the pane as tall as its tallest possible text, and pin it.

        Both bounds are set: a MINIMUM so nothing clips, and a MAXIMUM so the
        pane does not grow when a short description replaces a long one, which
        is what would move the buttons out from under the pointer.

        The width is the pane's own once it has one, and the dialog's hint
        before that -- at construction no layout has run, so `width()` is a
        placeholder and measuring against it would size for a pane one pixel
        wide.
        """
        width = self._measurement_width()
        metrics = QFontMetrics(self._description.font())
        tallest = 0
        for text in self.every_description():
            box = metrics.boundingRect(
                QRect(0, 0, width, 0),
                int(Qt.TextWordWrap | Qt.AlignTop | Qt.AlignLeft),
                text)
            tallest = max(tallest, box.height())
        tallest += metrics.lineSpacing()
        self._description.setMinimumHeight(tallest)
        self._description.setMaximumHeight(tallest)

    def _measurement_width(self) -> int:
        """The width the pane will actually be drawn at.

        Its own, ONCE A LAYOUT HAS RUN, and not before -- which is the whole
        fix. A freshly constructed QWidget reports 100 px, a number Qt gives
        every widget before anything sizes it and which this pane is never
        drawn at. The old guard only rejected widths of 1 or less, so it
        took the 100, wrapped 316 characters into a 100 px column, and
        reserved 425 px of height for a pane that needs 119. That is where a
        509 px dialog came from: not from the text, from measuring the text
        against a placeholder.

        Before the layout runs, the dialog's INTENDED width minus the
        layout's margins. Not `sizeHint().width()` either -- at construction
        that is whatever the two short buttons need, which is narrower still.
        """
        if self._laid_out and self._description.width() > 1:
            return self._description.width()
        layout = self.layout()
        margins = layout.contentsMargins() if layout is not None else None
        inset = (margins.left() + margins.right()) if margins is not None else 0
        return max(self.DIALOG_WIDTH - inset, 240)

    def resizeEvent(self, event):  # noqa: N802 - Qt name
        """Re-measure when the dialog is resized.

        A wider pane needs fewer lines and a narrower one needs more, so a
        height measured at one width clips at another. The user can resize
        this dialog, so this is reachable.
        """
        super().resizeEvent(event)
        self._laid_out = True
        self._size_the_description_pane()

    def _choose(self, key: str) -> None:
        """Record the chosen dataset and accept the dialog.

        :param key: the dataset the user picked.
        """
        self.chosen = str(key)
        self.accept()



__all__ = ["TestDataChooser"]
