"""The tile that appears when a gene is clicked in the interactive regression.

A thin renderer, deliberately: everything that can be wrong
about which gene a dot names lives in :mod:`spacr.gene_tile`, which is a pure
function over the ``feature`` string and the results frame and is tested
without a window. This module lays that record out and opens the links.

There are two entry points:

* :class:`GeneTilePanel` — a widget to sit beside the volcano and the results
  table, wired to the ``key_selected`` signal they both emit.
* :meth:`GeneTilePanel.to_pixmap` — the same record rendered to a ``QPixmap``,
  so the tile can go into the figure grid as "the same pressable tile every
  figure is, holding text instead of a plot" WITHOUT ``_FigureCell`` learning
  about text. The grid takes pixmaps; this gives it one.

The links are opened on click and never followed on render. Building a ToxoDB
URL is string formatting; a tile that fetched it would put a network round
trip inside a mouse click.
"""
from __future__ import annotations

import html
import logging
from typing import Callable, Optional

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy, QTextBrowser, QVBoxLayout, QWidget

from ...gene_tile import GeneTile, _translated, gene_tile
from ..i18n import tr
from ..theme import SPACING, font_px

LOG = logging.getLogger("spacr.qt.gene_tile")

__all__ = ["GeneTilePanel"]

#: What the tile says before anything has been clicked. Not blank: a blank
#: panel beside a plot reads as a panel that is broken rather than as one
#: waiting for a selection.
IDLE_TEXT = ("Click a point in the volcano, or a row in the results table, "
             "to see what that gene is.")

#: The pixmap the figure grid gets is drawn at this width unless told another.
#: Matches the grid's own MIN_CELL_PX so a gene tile does not resize the row
#: it lands in.
TILE_WIDTH = 220


class GeneTilePanel(QWidget):
    """Everything spaCR knows about the clicked gene, laid out.

    :param frame_provider: called with no arguments to get the current results
        frame. A callable rather than a stored frame so the panel cannot go on
        answering from the previous regression after a new one is loaded —
        which is the bug shape this whole cluster keeps producing.
    :param parent: the usual.
    """

    #: Emitted with the feature string whenever a tile is built for it, so a
    #: host can log or mirror the selection.
    tile_shown = Signal(str)

    def __init__(self, frame_provider: Optional[Callable[[], object]] = None,
                 parent=None):
        super().__init__(parent)
        self._frame_provider = frame_provider
        self._tile: Optional[GeneTile] = None
        self._feature = ""
        self._error_feature = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.get("xs", 4))

        self._view = QTextBrowser()
        self._view.setOpenLinks(False)
        self._view.setOpenExternalLinks(False)
        # A gene id must survive translation intact: TGGT1_239740 is not a
        # phrase, and a catalog that "translated" it would be renaming a gene.
        self._view.setProperty("i18nSkipText", True)
        self._view.anchorClicked.connect(self._open)
        self._view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._view, 1)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(
            f"color: palette(mid); font-size: {font_px(10)}px;")
        layout.addWidget(self._status)

        self.clear()

    # ------------------------------------------------------------------ state

    @property
    def tile(self) -> Optional[GeneTile]:
        """The record currently shown, or ``None`` before the first click."""
        return self._tile

    @property
    def feature(self) -> str:
        """The feature string the current tile was built from."""
        return self._feature

    def clear(self) -> None:
        """Back to the waiting state, which says it is waiting."""
        self._tile = None
        self._feature = ""
        self._error_feature = ""
        self._render_content()

    def set_frame_provider(self, provider: Optional[Callable[[], object]]
                           ) -> None:
        """Point the panel at where the current results frame lives."""
        self._frame_provider = provider

    # ------------------------------------------------------------------ slots

    def show_feature(self, key: str) -> None:
        """Build and show the tile for one clicked feature.

        THE SLOT TO CONNECT ``key_selected`` TO. Takes the feature string and
        nothing else, so the volcano and the results table reach it
        identically.

        A failure here must not take the plot down with it: a tile is an
        explanation, and an explanation that raises leaves the user with a
        traceback instead of the point they clicked.
        """
        frame = None
        if self._frame_provider is not None:
            try:
                frame = self._frame_provider()
            except Exception:
                # A BROKEN HOST, NOT A BROKEN TILE. The provider belongs
                # to whatever screen owns this panel; if it raises, the
                # tile still draws from no frame rather than letting the
                # host's failure out through a click on a plot point.
                LOG.exception("gene tile: could not reach the results frame")
        try:
            tile = gene_tile(key, frame)
        except Exception:
            LOG.exception("gene tile: could not resolve %r", key)
            self._tile = None
            self._feature = str(key)
            self._error_feature = self._feature
            self._render_content()
            return

        self._tile = tile
        self._feature = tile.feature
        self._error_feature = ""
        self._render_content()
        self.tile_shown.emit(self._feature)

    def _render_content(self, language: Optional[str] = None) -> None:
        """Render application prose in ``language`` without changing data."""
        translate = lambda source, **values: tr(  # noqa: E731
            source, language, **values)
        if self._error_feature:
            message = translate(
                "Could not build a tile for {feature}. The plot is "
                "unaffected; see the log for details.",
                feature=self._error_feature,
            )
            self._view.setHtml(
                f"<p style='color:#c66'>{html.escape(message)}</p>")
            self._status.setText("")
            return
        if self._tile is None:
            self._view.setHtml(
                "<p style='color:#888'>"
                + html.escape(translate(IDLE_TEXT))
                + "</p>"
            )
            self._status.setText("")
            return
        self._view.setHtml(self._tile.to_html(translate))
        if self._tile.ambiguous:
            status = translate(
                "ambiguous mapping — every gene it could be is listed above"
            )
        elif self._tile.unresolved and not self._tile.resolved:
            status = _translated(self._tile.unresolved[0], translate)
        else:
            status = ""
        self._status.setText(status)

    def retranslate_dynamic_content(
        self,
        language: Optional[str] = None,
    ) -> None:
        """Refresh the structured tile after the application language changes."""
        self._render_content(language)

    def _open(self, url: QUrl) -> None:
        """Follow an external reference — on the click, never on the render."""
        QDesktopServices.openUrl(url)

    # ------------------------------------------------------------------ grid

    def to_pixmap(self, width: int = TILE_WIDTH) -> QPixmap:
        """The current tile as a ``QPixmap``, for the figure grid.

        The grid's cells take a pixmap and size themselves from its aspect
        ratio. Rendering the text to one lets the gene tile be a tile in that
        grid without the grid growing a second kind of cell.
        """
        from PySide6.QtGui import QTextDocument

        document = QTextDocument()
        document.setHtml(self._view.toHtml())
        document.setTextWidth(max(int(width), 1))
        size = document.size().toSize()
        pixmap = QPixmap(QSize(max(int(width), 1), max(size.height(), 1)))
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        try:
            document.drawContents(painter)
        finally:
            painter.end()
        return pixmap
