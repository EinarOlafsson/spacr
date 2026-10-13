"""Viewer for the 3-page model cards of the Model Zoo.

Shows the page image of a card scaled to fit, with Previous and Next, a
"1 / 3" counter, Left and Right keys and a link to the model on Hugging Face.
The images are fetched on first view in a worker thread and cached; without
them the viewer shows the same three pages as text from the card data that
ships with spaCR. Clicking the image opens it larger in a dialog.
"""
from __future__ import annotations

import html
import threading
from typing import Any, Callable, Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel,
                               QPushButton, QSizePolicy, QStackedLayout,
                               QTextBrowser, QVBoxLayout)

from ... import model_cards as cards
from ..hidpi import scaled_for
from ..i18n import tr
from ..theme import RADIUS, SPACING, active_palette

VIEWER_NAME = "ModelCardViewer"


class _PageLabel(QLabel):
    """A label that keeps its pixmap scaled to its size, aspect kept."""

    clicked = Signal()

    def __init__(self, parent=None):
        """An empty label, centred, that any layout may resize.

        :param parent: parent widget.
        """
        super().__init__(parent)
        self._source: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(160, 120)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

    def set_source(self, pixmap: Optional[QPixmap]) -> None:
        """Show ``pixmap`` (full resolution) scaled to fit."""
        self._source = pixmap
        self._rescale()

    def _rescale(self) -> None:
        """Fit the source pixmap to the label's current size."""
        if self._source is None or self._source.isNull():
            self.clear()
            return
        self.setPixmap(scaled_for(self._source, self,
                                  max(1, self.width()), max(1, self.height())))

    def resizeEvent(self, event):  # noqa: N802
        """Refit the pixmap to the new size.

        :param event: the Qt resize event.
        """
        super().resizeEvent(event)
        self._rescale()

    def mouseReleaseEvent(self, event):  # noqa: N802
        """Emit :attr:`clicked` on a left click.

        :param event: the Qt mouse event.
        """
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class ModelCardViewer(QFrame):
    """Page through the card of one model.

    :param parent: parent widget.
    """

    #: Emitted with the zero-based page after every page change.
    page_changed = Signal(int)
    _fetched = Signal(str, object)

    def __init__(self, parent=None, *, threaded: bool = True,
                 fetch: Optional[Callable[[str], bytes]] = None,
                 enlargeable: bool = True):
        """:param threaded: fetch the images on a worker thread; ``False`` fetches
            inline, for tests.
        :param fetch: byte source taking a URL, handed to
            :func:`spacr.model_cards.fetch_pages`; ``None`` uses the network.
        :param enlargeable: clicking the image opens a larger dialog.
        """
        super().__init__(parent)
        self.setObjectName(VIEWER_NAME)
        self.setFocusPolicy(Qt.StrongFocus)
        self._threaded = threaded
        self._fetch = fetch
        self._enlargeable = enlargeable
        self._card: Optional[Dict[str, Any]] = None
        self._pixmaps: List[Optional[QPixmap]] = [None] * cards.PAGE_COUNT
        self._page = 0
        self._failed = False
        self._dialog: Optional[QDialog] = None
        self._thread: Optional[threading.Thread] = None
        self._fetched.connect(self._on_fetched)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACING["xs"], SPACING["xs"],
                                 SPACING["xs"], SPACING["xs"])
        outer.setSpacing(SPACING["xs"])

        self._stack = QStackedLayout()
        self._image = _PageLabel(self)
        self._image.clicked.connect(self._enlarge)
        self._text = QTextBrowser(self)
        self._text.setObjectName("ModelCardText")
        self._stack.addWidget(self._image)
        self._stack.addWidget(self._text)
        outer.addLayout(self._stack, 1)

        self._status = QLabel("", self)
        self._status.setObjectName("Muted")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        row = QHBoxLayout()
        row.setSpacing(SPACING["sm"])
        row.setContentsMargins(0, 0, 0, 0)
        self._prev = QPushButton(tr("Previous"), self)
        self._next = QPushButton(tr("Next"), self)
        for button in (self._prev, self._next):
            button.setFixedHeight(26)
        self._prev.clicked.connect(self.previous_page)
        self._next.clicked.connect(self.next_page)
        self._counter = QLabel("", self)
        self._counter.setAlignment(Qt.AlignCenter)
        self._link = QLabel("", self)
        self._link.setOpenExternalLinks(True)
        self._link.setTextInteractionFlags(Qt.TextBrowserInteraction)
        for button in (self._prev, self._next):
            button.setFocusPolicy(Qt.NoFocus)
        row.addWidget(self._prev)
        row.addWidget(self._counter, 1)
        row.addWidget(self._next)
        row.addSpacing(SPACING["sm"])
        row.addWidget(self._link)
        outer.addLayout(row)
        self._apply_style()
        self._show_page()

    def card(self) -> Optional[Dict[str, Any]]:
        """The card data being shown, or ``None``."""
        return self._card

    def page(self) -> int:
        """The zero-based current page."""
        return self._page

    def counter_text(self) -> str:
        """The "1 / 3" text."""
        return self._counter.text()

    def showing_image(self) -> bool:
        """True when the current page is the picture, not the text."""
        return self._stack.currentWidget() is self._image

    def status_text(self) -> str:
        """The line under the page."""
        return self._status.text()

    def set_card(self, card: Optional[Dict[str, Any]]) -> None:
        """Show ``card`` from page 1, fetching its images when not cached.

        :param card: the card data, or None to show nothing.
        """
        self._card = card
        self._page = 0
        self._failed = False
        self._pixmaps = [None] * cards.PAGE_COUNT
        if card is None:
            self._show_page()
            return
        repo = str(card["repo_id"])
        cached = cards.cached_pages(repo)
        if all(cached):
            self._load(cached)
        else:
            self._status.setText(tr("Loading the card images…"))
            self._start_fetch(repo)
        self._show_page()

    def go_to(self, page: int) -> None:
        """Show page ``page`` (zero-based), clamped to the pages.

        :param page: the page index, from 0.
        """
        page = max(0, min(cards.PAGE_COUNT - 1, int(page)))
        if page != self._page:
            self._page = page
            self._show_page()
            self.page_changed.emit(page)

    def next_page(self) -> None:
        """Show the next page; stays on the last."""
        self.go_to(self._page + 1)

    def previous_page(self) -> None:
        """Show the previous page; stays on the first."""
        self.go_to(self._page - 1)

    def keyPressEvent(self, event):  # noqa: N802
        """Left and Right turn the page.

        :param event: the Qt key event.
        """
        if event.key() == Qt.Key_Right:
            self.next_page()
        elif event.key() == Qt.Key_Left:
            self.previous_page()
        else:
            super().keyPressEvent(event)

    def _start_fetch(self, repo: str) -> None:
        """Fetch the page images of ``repo``, on a worker when threaded.

        :param repo: ``owner/repo`` of the model.
        """
        def job():
            """Fetch the card pages and emit the outcome, or the failure."""
            try:
                result: Any = cards.fetch_pages(repo, self._fetch)
            except Exception as exc:  # noqa: BLE001 - any failure is offline
                result = exc
            try:
                self._fetched.emit(repo, result)
            except RuntimeError:
                return

        if not self._threaded:
            job()
            return
        self._thread = threading.Thread(target=job, daemon=True)
        self._thread.start()

    def _on_fetched(self, repo: str, result: Any) -> None:
        """Take the fetched pages, or note the failure, for the shown card.

        :param repo: the repo the fetch was for; a stale one is ignored.
        :param result: the page paths, or the exception the fetch raised.
        """
        if not self._card or self._card.get("repo_id") != repo:
            return
        if isinstance(result, BaseException):
            self._failed = True
        else:
            self._load(list(result))
        self._show_page()

    def _load(self, paths) -> None:
        """Read the page images; one that does not load fails them all.

        :param paths: the page image files, in page order.
        """
        self._pixmaps = [QPixmap(str(p)) for p in paths]
        if any(p.isNull() for p in self._pixmaps):
            self._pixmaps = [None] * cards.PAGE_COUNT
            self._failed = True
        else:
            self._failed = False

    def _show_page(self) -> None:
        """Show the current page: its image, else its text from the card."""
        card = self._card
        self._prev.setEnabled(card is not None and self._page > 0)
        self._next.setEnabled(
            card is not None and self._page < cards.PAGE_COUNT - 1)
        dots = " ".join("●" if i == self._page else "○"
                        for i in range(cards.PAGE_COUNT))
        self._counter.setText(
            f"{dots}   {self._page + 1} / {cards.PAGE_COUNT}"
            if card else "")
        if card is None:
            self._link.setText("")
            self._image.set_source(None)
            self._text.setHtml("")
            self._stack.setCurrentWidget(self._text)
            return
        self._link.setText(
            f'<a href="{cards.model_link(card)}" '
            f'style="color:{active_palette()["accent"]}">'
            f'{tr("Open on Hugging Face")}</a>')
        pixmap = self._pixmaps[self._page]
        if pixmap is not None:
            self._image.set_source(pixmap)
            self._stack.setCurrentWidget(self._image)
            self._status.setText("")
            self._image.setCursor(
                Qt.PointingHandCursor if self._enlargeable
                else Qt.ArrowCursor)
            self._image.setToolTip(
                tr("Click to enlarge") if self._enlargeable else "")
            return
        if self._failed:
            self._status.setText(
                tr("The card images could not be downloaded. Showing the "
                   "card data stored with spaCR."))
        headers = (tr("Score"), tr("This model (stock base)"),
                   tr("Fact"), tr("Value"))
        palette = active_palette()
        self._text.setHtml(
            f"<body style='color:{palette['fg']}'>"
            f"<h3>{html.escape(str(card.get('display_name', '')))}</h3>"
            + cards.page_html(card, self._page + 1, headers) + "</body>")
        self._stack.setCurrentWidget(self._text)

    def _apply_style(self) -> None:
        """Style the frame, labels and text from the active palette."""
        p = active_palette()
        self.setStyleSheet(
            f"QFrame#{VIEWER_NAME} {{ border: 1px solid {p['border_soft']};"
            f" border-radius: {RADIUS['md']}px; }}"
            f"QFrame#{VIEWER_NAME} QLabel {{ background: transparent;"
            f" color: {p['fg']}; }}"
            f"QTextBrowser#ModelCardText {{ background: transparent;"
            f" border: none; }}")

    def _enlarge(self) -> None:
        """Open the current page larger in a dialog of its own."""
        if not self._enlargeable or self._card is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(str(self._card.get("display_name", "")))
        layout = QVBoxLayout(dialog)
        big = ModelCardViewer(dialog, threaded=False, fetch=self._fetch,
                              enlargeable=False)
        layout.addWidget(big)
        big.set_card(self._card)
        big.go_to(self._page)
        dialog.resize(1000, 800)
        self._dialog = dialog
        dialog.show()
