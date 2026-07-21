"""
StartupPage — the home screen with a title, subtitle, and a grid of
clickable tiles for each spacr app.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ..theme import SPACING
from ..widgets import Divider, Tile


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

        # Hero header block
        eyebrow = QLabel("SpaCR")
        eyebrow.setObjectName("Caption")
        eyebrow.setAlignment(Qt.AlignHCenter)
        outer.addWidget(eyebrow)

        title = QLabel("Spatial phenotype analysis")
        title.setObjectName("Hero")
        title.setAlignment(Qt.AlignHCenter)
        outer.addWidget(title)

        subtitle = QLabel(
            "End-to-end microscopy → single-cell measurements → "
            "genotype–phenotype mapping. Pick an app to get started."
        )
        subtitle.setObjectName("SubtitleSmall")
        subtitle.setAlignment(Qt.AlignHCenter)
        subtitle.setWordWrap(True)
        subtitle.setMaximumWidth(720)
        subtitle_wrap = QHBoxLayout()
        subtitle_wrap.addStretch(1)
        subtitle_wrap.addWidget(subtitle)
        subtitle_wrap.addStretch(1)
        outer.addLayout(subtitle_wrap)

        outer.addSpacing(SPACING["xl"])

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
                tile = Tile(text=name, icon=icon, caption=name, tile_size=104, icon_size=46)
                tile.setToolTip(desc)
                tile.clicked.connect(lambda k=key: self.tile_clicked.emit(k))
                grid.addWidget(tile, i // cols, i % cols)

            outer.addWidget(grid_container)

        outer.addStretch(1)

        # Footer
        footer = QLabel("Click a tile to open an application, or use the sidebar.")
        footer.setObjectName("SubtitleSmall")
        footer.setAlignment(Qt.AlignHCenter)
        outer.addWidget(footer)

        self.setWidget(content)
