"""The tile that appears when a gene is clicked in the interactive regression.

Instruction 121. A thin renderer, deliberately: everything that can be WRONG
about which gene a dot names lives in :mod:`spacr.gene_tile`, which is a pure
function over the ``feature`` string and the results frame and is tested
without a window. This module lays that record out and opens the links.

Two ways in, because the instruction asks for both:

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

import logging
from typing import Callable, Optional

from PySide6.QtCore import QSize, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QPainter, QPixmap
from PySide6.QtWidgets import (QLabel, QSizePolicy, QTextBrowser, QVBoxLayout,
                               QWidget)

from ...gene_tile import GeneTile, gene_tile
from ..theme import SPACING

LOG = logging.getLogger("spacr.qt.gene_tile")

__all__ = ["GeneTilePanel"]

#: What the tile says before anything has been clicked. Not blank: a blank
#: panel beside a plot reads as a panel that is broken rather than as one
#: waiting, which is the same failure the instruction names for the tile
#: itself.
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
        self._status.setStyleSheet("color: palette(mid); font-size: 10px;")
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
        self._view.setHtml(f"<p style='color:#888'>{IDLE_TEXT}</p>")
        self._status.setText("")

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
            except Exception:  # pragma: no cover - a broken host, not a tile
                LOG.exception("gene tile: could not reach the results frame")
        try:
            tile = gene_tile(key, frame)
        except Exception:
            LOG.exception("gene tile: could not resolve %r", key)
            self._tile = None
            self._feature = str(key)
            self._view.setHtml(
                f"<p style='color:#c66'>Could not build a tile for "
                f"<b>{self._feature}</b>. The plot is unaffected; see the log "
                "for what went wrong.</p>")
            self._status.setText("")
            return

        self._tile = tile
        self._feature = tile.feature
        self._view.setHtml(tile.to_html())
        self._status.setText(
            "ambiguous mapping — every gene it could be is listed above"
            if tile.ambiguous else
            (tile.unresolved[0] if tile.unresolved and not tile.resolved
             else ""))
        self.tile_shown.emit(self._feature)

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
