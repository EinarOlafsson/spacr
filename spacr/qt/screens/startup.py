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

        # Title block
        title = QLabel("SpaCR")
        title.setObjectName("DisplayHeading")
        title.setAlignment(Qt.AlignHCenter)
        outer.addWidget(title)

        subtitle = QLabel("Spatial single-cell analysis for microscopy")
        subtitle.setObjectName("Subtitle")
        subtitle.setAlignment(Qt.AlignHCenter)
        outer.addWidget(subtitle)

        outer.addSpacing(SPACING["md"])

        # Group apps by section, render each section as a heading + tile grid.
        sections: dict[str, list[tuple[str, str, str]]] = {}
        for key, name, desc, section in apps:
            sections.setdefault(section, []).append((key, name, desc))

        for section_name, entries in sections.items():
            hdr = QLabel(section_name.upper())
            hdr.setObjectName("SectionHeading")
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
                tile = Tile(text=name, icon=icon, caption=name, tile_size=96, icon_size=42)
                tile.setToolTip(desc)
                tile.clicked.connect(lambda k=key: self.tile_clicked.emit(k))
                grid.addWidget(tile, i // cols, i % cols)

            outer.addWidget(grid_container)

        outer.addStretch(1)

        # Footer
        footer = QLabel("Click a tile to open an application. "
                        "Use the sidebar for quick navigation.")
        footer.setObjectName("Muted")
        footer.setAlignment(Qt.AlignHCenter)
        outer.addWidget(footer)

        self.setWidget(content)
