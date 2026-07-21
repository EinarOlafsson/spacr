"""
StartupPage — the home screen with a title, subtitle, and a grid of
clickable tiles for each spacr app.
"""
from __future__ import annotations

import os
from typing import Callable, List, Tuple

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..theme import PALETTE, SPACING
from ..widgets import Divider, Tile


def _find_logo_pixmap() -> QPixmap | None:
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
    """Home screen. `tile_clicked(str key)` fires when a tile is pressed."""

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

        # Hero header block — logo + wordmark side by side, with subtitle
        # below on a full-width band.
        hero = QFrame()
        hero.setObjectName("Hero")
        hero_col = QVBoxLayout(hero)
        hero_col.setContentsMargins(SPACING["xl"], SPACING["xl"],
                                     SPACING["xl"], SPACING["xl"])
        hero_col.setSpacing(SPACING["md"])
        hero_col.setAlignment(Qt.AlignCenter)

        brand_row = QHBoxLayout()
        brand_row.setSpacing(SPACING["lg"])
        brand_row.setAlignment(Qt.AlignCenter)

        logo_pix = _find_logo_pixmap()
        if logo_pix is not None:
            logo_lbl = QLabel()
            logo_lbl.setPixmap(
                logo_pix.scaled(96, 96, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
            logo_lbl.setFixedSize(96, 96)
            logo_lbl.setAlignment(Qt.AlignCenter)
            brand_row.addWidget(logo_lbl)

        wordmark_col = QVBoxLayout()
        wordmark_col.setContentsMargins(0, 0, 0, 0)
        wordmark_col.setSpacing(2)
        wordmark_col.setAlignment(Qt.AlignVCenter)

        eyebrow = QLabel("SpaCR")
        eyebrow.setObjectName("Caption")
        wordmark_col.addWidget(eyebrow)

        title = QLabel("Spatial phenotype analysis")
        title.setObjectName("Hero")
        wordmark_col.addWidget(title)

        brand_row.addLayout(wordmark_col)
        hero_col.addLayout(brand_row)

        subtitle = QLabel(
            "End-to-end microscopy → single-cell measurements → "
            "genotype–phenotype mapping."
        )
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignHCenter)
        subtitle.setWordWrap(True)
        subtitle.setMaximumWidth(820)
        hero_col.addWidget(subtitle, alignment=Qt.AlignHCenter)

        outer.addWidget(hero)
        outer.addSpacing(SPACING["md"])

        # Group apps by section, render each section as a heading + tile grid.
        sections: dict[str, list[tuple[str, str, str]]] = {}
        for key, name, desc, section in apps:
            sections.setdefault(section, []).append((key, name, desc))

        for section_name, entries in sections.items():
            hdr = QLabel(section_name.upper())
            hdr.setObjectName("Caption")
            outer.addWidget(hdr)

            divider = Divider()
            outer.addWidget(divider)

            grid_container = QWidget()
            grid = QGridLayout(grid_container)
            grid.setContentsMargins(0, SPACING["sm"], 0, SPACING["sm"])
            grid.setHorizontalSpacing(SPACING["lg"])
            grid.setVerticalSpacing(SPACING["md"])
            grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)

            cols = 6
            for i, (key, name, desc) in enumerate(entries):
                icon = icon_provider(key)
                tile = Tile(text=name, icon=icon, caption=name,
                            tile_size=128, icon_size=68)
                tile.setToolTip(desc)
                tile.clicked.connect(lambda k=key: self.tile_clicked.emit(k))
                # Hover shows the description in the footer
                tile._button.installEventFilter(self)
                tile._button.setProperty("hover_caption", desc)
                grid.addWidget(tile, i // cols, i % cols)

            outer.addWidget(grid_container)

        outer.addStretch(1)

        # Hover-follows footer — replaced with dynamic caption on tile hover.
        self._hover_footer = QLabel(
            "Hover a tile to see what each app does."
        )
        self._hover_footer.setObjectName("SubtitleSmall")
        self._hover_footer.setAlignment(Qt.AlignHCenter)
        self._hover_footer.setMinimumHeight(28)
        outer.addWidget(self._hover_footer)

        self.setWidget(content)

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if event.type() == QEvent.Enter:
            cap = obj.property("hover_caption")
            if cap:
                self._hover_footer.setText(cap)
        elif event.type() == QEvent.Leave:
            self._hover_footer.setText(
                "Hover a tile to see what each app does."
            )
        return super().eventFilter(obj, event)
