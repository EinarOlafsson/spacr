"""StartupPage — the home screen.

Layout, top to bottom:

    ┌──────────────────────────────────────────────────────────┐
    │             spaCR  🖼    (wordmark + logo right)          │
    │        End-to-end microscopy → single-cell …            │
    ├──────────────────────────────────────────────────────────┤
    │  CORE  ──────────────────────────────────────            │
    │    🖼  Mask     — Cellpose segmentation of cells…         │
    │    📏  Measure  — Per-object feature extraction…         │
    │    …                                                    │
    │  ANALYSIS  ──────────────────────────────────            │
    │    …                                                    │
    └──────────────────────────────────────────────────────────┤
    │  Hover a tile to see what it does.                       │  ← sticky
    └──────────────────────────────────────────────────────────┘

The hint bar at the bottom stays pinned as the user scrolls so the
description of whatever's under the cursor is always visible — no
guessing what the tiny descriptions in the tiles mean.
"""
from __future__ import annotations

import os
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE, SPACING
from ..widgets import Divider, HTile


def _find_logo_pixmap() -> Optional[QPixmap]:
    """Locate the bundled spaCR logo, returning ``None`` if absent."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in ("logo_spacr.png", "logo_spacr_v1.png"):
        p = os.path.normpath(
            os.path.join(here, "..", "..", "resources", "icons", candidate)
        )
        if os.path.isfile(p):
            pix = QPixmap(p)
            if not pix.isNull():
                return pix
    return None


_DEFAULT_HINT = "Hover a tile to see what it does."


class StartupPage(QWidget):
    """Home screen. ``tile_clicked(str key)`` fires when a row is pressed.

    :param apps: ``(key, name, description, section)`` tuples describing
        each app to render.
    :param icon_provider: callable that returns a QIcon for a given app
        key (or ``None`` for a text-fallback tile).
    :param parent: optional parent widget.
    :ivar tile_clicked: emitted with the app key when a tile is pressed.
    """

    tile_clicked = Signal(str)

    def __init__(
        self,
        apps: List[Tuple[str, str, str, str]],
        icon_provider: Callable[[str], Optional[QIcon]],
        parent=None,
    ):
        super().__init__(parent)

        # Two-row outer layout: scrollable content + sticky hint bar.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        col = QVBoxLayout(content)
        col.setContentsMargins(SPACING["xxl"], SPACING["xxl"],
                                SPACING["xxl"], SPACING["xxl"])
        col.setSpacing(SPACING["xl"])

        col.addWidget(self._build_hero())

        # Sections — keep a mapping from every HTile to its description
        # so the sticky footer knows what to display on hover.
        self._tile_hints: dict = {}
        sections: dict[str, list[tuple[str, str, str]]] = {}
        for key, name, desc, section in apps:
            sections.setdefault(section, []).append((key, name, desc))

        for section_name, entries in sections.items():
            col.addWidget(self._build_section_header(section_name))
            col.addWidget(self._build_section_grid(entries, icon_provider))

        col.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        # Sticky hint bar — always visible along the bottom, updates
        # on tile hover, reverts to the default when the mouse leaves.
        self._hint_bar = QLabel(_DEFAULT_HINT)
        self._hint_bar.setObjectName("HintBar")
        self._hint_bar.setAlignment(Qt.AlignHCenter)
        self._hint_bar.setMinimumHeight(36)
        self._hint_bar.setStyleSheet(
            "background-color: "
            f"{PALETTE['surface_alt']};"
            f"border-top: 1px solid {PALETTE['border_soft']};"
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 300;"
            "font-size: 13px;"
            f"color: {PALETTE['fg_muted']};"
            "padding: 8px 12px;"
        )
        outer.addWidget(self._hint_bar)

    # -- pieces --------------------------------------------------------
    def _build_hero(self) -> QWidget:
        """Wordmark + logo (LEFT: text, RIGHT: logo), no border/frame.

        The old version wrapped everything in a rounded QFrame; that
        looked like a card on top of a card and the border was
        distracting. This is just the wordmark + subtitle on the
        page background — nothing to draw attention away from the
        app tiles below.
        """
        hero = QWidget()
        outer = QVBoxLayout(hero)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(SPACING["md"])
        outer.setAlignment(Qt.AlignCenter)

        # Row: [wordmark]  [logo]  — logo on the RIGHT of the text
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(SPACING["lg"])
        row.setAlignment(Qt.AlignCenter)

        title = QLabel("spaCR")
        title.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 300;"
            "font-size: 56px;"
            f"color: {PALETTE['accent']};"
            "letter-spacing: -1.2px;"
            "background: transparent;"
        )
        row.addWidget(title, alignment=Qt.AlignVCenter)

        logo_pix = _find_logo_pixmap()
        if logo_pix is not None:
            logo_lbl = QLabel()
            logo_lbl.setPixmap(
                logo_pix.scaled(72, 72, Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
            )
            logo_lbl.setFixedSize(72, 72)
            logo_lbl.setStyleSheet("background: transparent;")
            row.addWidget(logo_lbl, alignment=Qt.AlignVCenter)

        outer.addLayout(row)

        subtitle = QLabel(
            "End-to-end microscopy → single-cell measurements "
            "→ genotype‑phenotype mapping."
        )
        subtitle.setAlignment(Qt.AlignHCenter)
        subtitle.setWordWrap(True)
        subtitle.setMinimumWidth(560)
        subtitle.setMaximumWidth(900)
        subtitle.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 300;"
            "font-size: 15px;"
            f"color: {PALETTE['fg_muted']};"
            "background: transparent;"
        )
        outer.addWidget(subtitle, alignment=Qt.AlignHCenter)

        return hero

    def _build_section_header(self, name: str) -> QWidget:
        wrap = QWidget()
        row = QVBoxLayout(wrap)
        row.setContentsMargins(0, SPACING["sm"], 0, 0)
        row.setSpacing(6)

        hdr = QLabel(name.upper())
        hdr.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 600;"
            "font-size: 11px;"
            "letter-spacing: 2px;"
            f"color: {PALETTE['fg_muted']};"
        )
        row.addWidget(hdr)
        row.addWidget(Divider())
        return wrap

    def _build_section_grid(
        self,
        entries: list[tuple[str, str, str]],
        icon_provider: Callable[[str], Optional[QIcon]],
    ) -> QWidget:
        wrap = QWidget()
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, SPACING["xs"], 0, SPACING["md"])
        grid.setHorizontalSpacing(SPACING["md"])
        grid.setVerticalSpacing(6)

        cols = 2
        for i, (key, name, desc) in enumerate(entries):
            icon = icon_provider(key)
            tile = HTile(text=name, description=desc, icon=icon,
                          icon_size=36)
            tile.setMinimumWidth(280)
            # Remember description so the hint bar knows what to say
            # when the cursor enters this tile.
            self._tile_hints[tile] = desc
            tile.installEventFilter(self)
            tile.clicked.connect(lambda checked=False, k=key:
                                   self.tile_clicked.emit(k))
            grid.addWidget(tile, i // cols, i % cols)

        for c in range(cols):
            grid.setColumnStretch(c, 1)
        return wrap

    # -- sticky hint bar wiring ----------------------------------------
    def eventFilter(self, obj, event):
        """Update the pinned hint bar as the cursor enters/leaves tiles."""
        if event.type() == QEvent.Enter:
            hint = self._tile_hints.get(obj)
            if hint:
                self._hint_bar.setText(hint)
        elif event.type() == QEvent.Leave:
            self._hint_bar.setText(_DEFAULT_HINT)
        return super().eventFilter(obj, event)
