"""A popup that explains each test-data route before either is started.

Instruction of 2026-09-01, for Annotate and then Classify: one "Load test
data" button opening a dialog with two buttons, a description that fills in on
hover, and a Close button.

It lives here rather than on a screen because two screens now use it and a
third is asked for. It fetches NOTHING itself -- it reports the chosen route
and the screen does the work, which keeps it testable without a network.
"""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QDialog,
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
         "route then works. Image source is set to STREAM IMAGES."),
    )

    #: What the description pane says before anything is hovered.
    RESTING_TEXT = ("Hover a button to see what it downloads and what it "
                    "sets. Nothing is fetched until you press one.")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Load test data"))
        self.chosen = ""

        layout = QVBoxLayout(self)

        buttons = QHBoxLayout()
        self._buttons = {}
        for key, label, description in self.ROUTES:
            button = QPushButton(tr(label), self)
            button.setCursor(Qt.PointingHandCursor)
            button.setProperty("routeKey", key)
            # The tooltip stays as well as the pane. The pane is the better
            # surface, but a tooltip is what a user reaches for by habit and
            # what the accessibility tree reads.
            button.setToolTip(description)
            button.installEventFilter(self)
            button.clicked.connect(
                lambda checked=False, k=key: self._choose(k))
            buttons.addWidget(button)
            self._buttons[key] = button
        layout.addLayout(buttons)

        self._description = QLabel(self.RESTING_TEXT, self)
        self._description.setWordWrap(True)
        self._description.setObjectName("TestDataDescription")
        # A FIXED HEIGHT, so the dialog does not resize under the pointer as
        # the text changes length -- the buttons would move away from the
        # cursor that is hovering them.
        self._description.setMinimumHeight(110)
        self._description.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self._description)

        closing = QHBoxLayout()
        closing.addStretch(1)
        close = QPushButton(tr("Close"), self)
        close.clicked.connect(self.reject)
        closing.addWidget(close)
        layout.addLayout(closing)

    def eventFilter(self, watched, event):      # noqa: N802 - Qt naming
        """Fill the pane on hover, and empty it on leave."""
        kind = event.type()
        if kind == QEvent.Enter:
            key = str(watched.property("routeKey") or "")
            for route_key, _label, description in self.ROUTES:
                if route_key == key:
                    self._description.setText(description)
                    break
        elif kind == QEvent.Leave:
            self._description.setText(self.RESTING_TEXT)
        return super().eventFilter(watched, event)

    def description_text(self) -> str:
        """What the pane currently says. For tests and the accessibility tree."""
        return self._description.text()

    def _choose(self, key: str) -> None:
        self.chosen = str(key)
        self.accept()



__all__ = ["TestDataChooser"]
