"""StartupPage — the home screen.

Minimalist layout:

  ┌────────────────── logo + wordmark ──────────────────┐
  │              spaCR — subtitle                       │
  └─────────────────────────────────────────────────────┘

  CORE ─────────────────────────────────────────
    🖼  Mask         — Cellpose segmentation of cells…
    📏  Measure      — Per-object feature extraction…
    …

Each app is a horizontal card (:class:`HTile`): icon on the left,
name on top in Open Sans Regular, one-line description underneath
in Open Sans Light. Nothing else — no borders unless you hover.
"""
from __future__ import annotations

import os
from typing import Callable, List, Tuple

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE, SPACING
from ..widgets import Divider, HTile


def _find_logo_pixmap() -> QPixmap | None:
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


class StartupPage(QScrollArea):
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
        icon_provider: Callable[[str], QIcon | None],
        parent=None,
    ):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        outer = QVBoxLayout(content)
        outer.setContentsMargins(SPACING["xxl"], SPACING["xxl"],
                                  SPACING["xxl"], SPACING["xxl"])
        outer.setSpacing(SPACING["xl"])

        # ─── Hero ────────────────────────────────────────────────────
        outer.addWidget(self._build_hero())

        # ─── App sections ────────────────────────────────────────────
        sections: dict[str, list[tuple[str, str, str]]] = {}
        for key, name, desc, section in apps:
            sections.setdefault(section, []).append((key, name, desc))

        for section_name, entries in sections.items():
            outer.addWidget(self._build_section_header(section_name))
            outer.addWidget(self._build_section_grid(entries, icon_provider))

        outer.addStretch(1)
        self.setWidget(content)

    # -- pieces --------------------------------------------------------
    def _build_hero(self) -> QWidget:
        hero = QFrame()
        hero.setObjectName("Hero")
        col = QVBoxLayout(hero)
        col.setContentsMargins(SPACING["xl"], SPACING["xl"],
                                SPACING["xl"], SPACING["xl"])
        col.setSpacing(SPACING["md"])
        col.setAlignment(Qt.AlignCenter)

        logo_pix = _find_logo_pixmap()
        if logo_pix is not None:
            logo_lbl = QLabel()
            logo_lbl.setPixmap(
                logo_pix.scaled(96, 96, Qt.KeepAspectRatio,
                                Qt.SmoothTransformation)
            )
            logo_lbl.setFixedSize(96, 96)
            logo_lbl.setAlignment(Qt.AlignCenter)
            col.addWidget(logo_lbl, alignment=Qt.AlignHCenter)

        title = QLabel("spaCR")
        title.setObjectName("Hero")
        title.setAlignment(Qt.AlignHCenter)
        title.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 300;"        # light for the wordmark
            "font-size: 44px;"
            "letter-spacing: -1px;"
        )
        col.addWidget(title, alignment=Qt.AlignHCenter)

        subtitle = QLabel(
            "End-to-end microscopy → single-cell measurements → "
            "genotype–phenotype mapping."
        )
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignHCenter)
        subtitle.setWordWrap(True)
        subtitle.setMaximumWidth(760)
        subtitle.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 300;"
            f"color: {PALETTE['fg_muted']};"
        )
        col.addWidget(subtitle, alignment=Qt.AlignHCenter)

        return hero

    def _build_section_header(self, name: str) -> QWidget:
        wrap = QWidget()
        row = QVBoxLayout(wrap)
        row.setContentsMargins(0, SPACING["sm"], 0, 0)
        row.setSpacing(6)

        hdr = QLabel(name.upper())
        hdr.setStyleSheet(
            "font-family: 'Open Sans', sans-serif;"
            "font-weight: 600;"        # semibold — quiet emphasis
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
        icon_provider: Callable[[str], QIcon | None],
    ) -> QWidget:
        wrap = QWidget()
        grid = QGridLayout(wrap)
        grid.setContentsMargins(0, SPACING["xs"], 0, SPACING["md"])
        grid.setHorizontalSpacing(SPACING["md"])
        grid.setVerticalSpacing(6)

        cols = 2   # icons-left layout wants wider rows, not a 6-wide grid
        for i, (key, name, desc) in enumerate(entries):
            icon = icon_provider(key)
            tile = HTile(text=name, description=desc, icon=icon,
                          icon_size=36)
            tile.setMinimumWidth(280)
            tile.clicked.connect(lambda checked=False, k=key:
                                   self.tile_clicked.emit(k))
            grid.addWidget(tile, i // cols, i % cols)

        # Uniform column widths
        for c in range(cols):
            grid.setColumnStretch(c, 1)
        return wrap
